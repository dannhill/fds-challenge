// main.c
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <direct.h> // Per _mkdir

// =============================================================================
// MODIFICA 1: Includere l'header di Windows per i thread e usare gli Intrinsics
// =============================================================================
#include <windows.h>      // Per l'API dei thread di Windows
#include <immintrin.h>    // Per istruzioni AVX2 (Intel Intrinsics)

// Librerie esterne
#include "libs/cJSON/cJSON.h"
#include "libs/uthash/uthash.h"

// =============================================================================
// 1. DEFINIZIONE COSTANTI E STRUTTURE DATI
//    (equivalente alla classe PokemonBattleModel e alle liste globali)
// =============================================================================

// Dimensioni delle feature, prese direttamente dallo script Python
#define POKEMON_NAMES_COUNT 151
#define MOVE_NAMES_COUNT 165
#define STATUSES_COUNT 8
#define EFFECTS_COUNT 2
#define TYPES_COUNT 20
#define MOVE_CATEGORIES_COUNT 3
#define BOOSTS_COUNT 6 // Python ne usava 5, ma il modello ne definisce 6. Usiamo 6 per coerenza col modello.

// Dimensione del vettore per un singolo Pokémon (squadra/lead)
#define POKEMON_STATIC_FEATURES_SIZE (POKEMON_NAMES_COUNT + 1 + (TYPES_COUNT * 2) + 6)

// Dimensione del vettore per lo stato di un giocatore in un turno
#define PLAYER_TURN_STATE_SIZE (POKEMON_NAMES_COUNT + 1 + STATUSES_COUNT + EFFECTS_COUNT + BOOSTS_COUNT + \
                                MOVE_NAMES_COUNT + TYPES_COUNT + MOVE_CATEGORIES_COUNT + 3)

// Dimensione totale del feature vector (calcolata come nello script Python)
// 6 P1 pokemons + 1 P2 lead + 30 turns * (P1 state + P2 state)
#define TOTAL_FEATURES ( (6 * POKEMON_STATIC_FEATURES_SIZE) + \
                         (1 * POKEMON_STATIC_FEATURES_SIZE) + \
                         (30 * (PLAYER_TURN_STATE_SIZE + PLAYER_TURN_STATE_SIZE)) )

// Struttura per la nostra hash map (per lookups veloci)
typedef struct {
    char* name;
    int index;
    UT_hash_handle hh;
} mapping_t;

// Mappe globali per i dati di gioco
mapping_t *pokemon_map = NULL, *move_map = NULL, *status_map = NULL, *type_map = NULL;

// =============================================================================
// 2. FUNZIONI DI SUPPORTO E UTILITY
// =============================================================================

// Funzione per caricare le chiavi di un file JSON in una hash map
void load_map_from_json(const char* filepath, mapping_t** map) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        perror("Error opening JSON file");
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* content = (char*)malloc(length + 1);
    fread(content, 1, length, f);
    fclose(f);
    content[length] = '\0';

    cJSON* json = cJSON_Parse(content);
    if (!json) {
        printf("Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(content);
        exit(1);
    }

    cJSON* child = json->child;
    int index = 0;
    while (child) {
        mapping_t* item = (mapping_t*)malloc(sizeof(mapping_t));
        item->name = _strdup(child->string); // _strdup per Windows
        item->index = index++;
        HASH_ADD_KEYPTR(hh, *map, item->name, strlen(item->name), item);
        child = child->next;
    }
    cJSON_Delete(json);
    free(content);
}

// Funzione per trovare l'indice in una mappa (case-insensitive opzionale)
int get_index_from_map(mapping_t* map, const char* name, bool case_insensitive) {
    mapping_t* item;
    if (case_insensitive) {
        char* lower_name = _strdup(name);
        for (char* p = lower_name; *p; ++p) *p = tolower(*p);
        HASH_FIND_STR(map, lower_name, item);
        free(lower_name);
    } else {
        HASH_FIND_STR(map, name, item);
    }
    return item ? item->index : -1;
}

// Funzione per la standardizzazione del vettore usando AVX2
// Questa è una delle "super ottimizzazioni"
void standardize_vector_avx(float* vector, int size) {
    double sum = 0.0, sq_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += vector[i];
        sq_sum += vector[i] * vector[i];
    }
    double mean = sum / size;
    double variance = (sq_sum / size) - (mean * mean);
    double stddev = (variance > 0) ? sqrt(variance) : 1.0;

    // Vettorizzazione con AVX2
    __m256 mean_vec = _mm256_set1_ps((float)mean);
    __m256 stddev_vec = _mm256_set1_ps((float)stddev);
    
    int i = 0;
    // Processa 8 float alla volta
    for (; i <= size - 8; i += 8) {
        __m256 data_vec = _mm256_loadu_ps(&vector[i]);
        data_vec = _mm256_sub_ps(data_vec, mean_vec);
        data_vec = _mm256_div_ps(data_vec, stddev_vec);
        _mm256_storeu_ps(&vector[i], data_vec);
    }

    // Processa gli elementi rimanenti
    for (; i < size; ++i) {
        vector[i] = (vector[i] - (float)mean) / (float)stddev;
    }
}


// =============================================================================
// 3. LOGICA DI ESTRAZIONE DELLE FEATURE (il cuore della traduzione)
// =============================================================================

// Funzione per processare i dettagli statici di un Pokémon
float* process_pokemon_details(cJSON* pokemon_data, float* feature_ptr) {
    if (!pokemon_data) {
        memset(feature_ptr, 0, POKEMON_STATIC_FEATURES_SIZE * sizeof(float));
        return feature_ptr + POKEMON_STATIC_FEATURES_SIZE;
    }

    // Name (one-hot)
    cJSON* name_node = cJSON_GetObjectItem(pokemon_data, "name");
    memset(feature_ptr, 0, POKEMON_NAMES_COUNT * sizeof(float));
    if (name_node && name_node->valuestring) {
        int idx = get_index_from_map(pokemon_map, name_node->valuestring, false);
        if (idx != -1) feature_ptr[idx] = 1.0f;
    }
    feature_ptr += POKEMON_NAMES_COUNT;

    // Level
    cJSON* level_node = cJSON_GetObjectItem(pokemon_data, "level");
    *feature_ptr++ = level_node ? (float)level_node->valueint : 0.0f;

    // Types (one-hot, max 2)
    cJSON* types = cJSON_GetObjectItem(pokemon_data, "types");
    memset(feature_ptr, 0, (TYPES_COUNT * 2) * sizeof(float));
    if (types) {
        int type_count = cJSON_GetArraySize(types);
        for (int i = 0; i < type_count && i < 2; ++i) {
            cJSON* type_item = cJSON_GetArrayItem(types, i);
            if (type_item && type_item->valuestring) {
                int idx = get_index_from_map(type_map, type_item->valuestring, true);
                if (idx != -1) feature_ptr[idx + i * TYPES_COUNT] = 1.0f;
            }
        }
    }
    feature_ptr += (TYPES_COUNT * 2);

    // Base Stats
    cJSON* stat_node;
    stat_node = cJSON_GetObjectItem(pokemon_data, "base_hp");
    *feature_ptr++ = stat_node ? (float)stat_node->valueint : 0.0f;
    stat_node = cJSON_GetObjectItem(pokemon_data, "base_atk");
    *feature_ptr++ = stat_node ? (float)stat_node->valueint : 0.0f;
    stat_node = cJSON_GetObjectItem(pokemon_data, "base_def");
    *feature_ptr++ = stat_node ? (float)stat_node->valueint : 0.0f;
    stat_node = cJSON_GetObjectItem(pokemon_data, "base_spa");
    *feature_ptr++ = stat_node ? (float)stat_node->valueint : 0.0f;
    stat_node = cJSON_GetObjectItem(pokemon_data, "base_spd");
    *feature_ptr++ = stat_node ? (float)stat_node->valueint : 0.0f;
    stat_node = cJSON_GetObjectItem(pokemon_data, "base_spe");
    *feature_ptr++ = stat_node ? (float)stat_node->valueint : 0.0f;

    return feature_ptr;
}

// Funzione per processare lo stato di un giocatore in un turno
float* process_player_turn_state(cJSON* turn_data, const char* player_prefix, float* feature_ptr) {
    char state_key[32], move_key[32];
    sprintf(state_key, "%s_pokemon_state", player_prefix);
    sprintf(move_key, "%s_move_details", player_prefix);

    cJSON* pokemon_state = cJSON_GetObjectItem(turn_data, state_key);
    if (!pokemon_state) {
        memset(feature_ptr, 0, PLAYER_TURN_STATE_SIZE * sizeof(float));
        return feature_ptr + PLAYER_TURN_STATE_SIZE;
    }

    // Name (one-hot)
    cJSON* name_node = cJSON_GetObjectItem(pokemon_state, "name");
    memset(feature_ptr, 0, POKEMON_NAMES_COUNT * sizeof(float));
    if (name_node && name_node->valuestring) {
        int idx = get_index_from_map(pokemon_map, name_node->valuestring, false);
        if (idx != -1) feature_ptr[idx] = 1.0f;
    }
    feature_ptr += POKEMON_NAMES_COUNT;

    // HP Pct
    cJSON* hp_node = cJSON_GetObjectItem(pokemon_state, "hp_pct");
    *feature_ptr++ = hp_node ? (float)hp_node->valuedouble : 0.0f;

    // Status (one-hot)
    cJSON* status_node = cJSON_GetObjectItem(pokemon_state, "status");
    memset(feature_ptr, 0, STATUSES_COUNT * sizeof(float));
    if (status_node && status_node->valuestring) {
        int idx = get_index_from_map(status_map, status_node->valuestring, false);
        if (idx != -1) feature_ptr[idx] = 1.0f;
    }
    feature_ptr += STATUSES_COUNT;

    // Effects (multi-hot)
    memset(feature_ptr, 0, EFFECTS_COUNT * sizeof(float));
    feature_ptr += EFFECTS_COUNT;

    // Boosts
    cJSON* boosts = cJSON_GetObjectItem(pokemon_state, "boosts");
    const char* boost_names[] = {"atk", "def", "spa", "spd", "spe", "accuracy"};
    for (int i = 0; i < BOOSTS_COUNT; ++i) {
        cJSON* boost_val = boosts ? cJSON_GetObjectItem(boosts, boost_names[i]) : NULL;
        *feature_ptr++ = boost_val ? (float)boost_val->valueint : 0.0f;
    }

    // Move Details
    cJSON* move_details = cJSON_GetObjectItem(turn_data, move_key);
    if (!move_details) {
        memset(feature_ptr, 0, (MOVE_NAMES_COUNT + TYPES_COUNT + MOVE_CATEGORIES_COUNT + 3) * sizeof(float));
        feature_ptr += (MOVE_NAMES_COUNT + TYPES_COUNT + MOVE_CATEGORIES_COUNT + 3);
    } else {
        // Move Name (one-hot)
        cJSON* move_name_node = cJSON_GetObjectItem(move_details, "name");
        memset(feature_ptr, 0, MOVE_NAMES_COUNT * sizeof(float));
        if (move_name_node && move_name_node->valuestring) {
            int idx = get_index_from_map(move_map, move_name_node->valuestring, false);
            if (idx != -1) feature_ptr[idx] = 1.0f;
        }
        feature_ptr += MOVE_NAMES_COUNT;

        // Move Type (one-hot)
        cJSON* move_type_node = cJSON_GetObjectItem(move_details, "type");
        memset(feature_ptr, 0, TYPES_COUNT * sizeof(float));
        if (move_type_node && move_type_node->valuestring) {
            int idx = get_index_from_map(type_map, move_type_node->valuestring, true);
            if (idx != -1) feature_ptr[idx] = 1.0f;
        }
        feature_ptr += TYPES_COUNT;

        // Move Category (one-hot)
        cJSON* cat_node = cJSON_GetObjectItem(move_details, "category");
        memset(feature_ptr, 0, MOVE_CATEGORIES_COUNT * sizeof(float));
        if (cat_node && cat_node->valuestring) {
            const char* cat_str = cat_node->valuestring;
            if (strcmp(cat_str, "PHYSICAL") == 0) feature_ptr[0] = 1.0f;
            else if (strcmp(cat_str, "SPECIAL") == 0) feature_ptr[1] = 1.0f;
            else if (strcmp(cat_str, "STATUS") == 0) feature_ptr[2] = 1.0f;
        }
        feature_ptr += MOVE_CATEGORIES_COUNT;

        // Move Stats
        cJSON* stat_val;
        stat_val = cJSON_GetObjectItem(move_details, "base_power");
        *feature_ptr++ = stat_val ? (float)stat_val->valueint : 0.0f;
        stat_val = cJSON_GetObjectItem(move_details, "accuracy");
        *feature_ptr++ = stat_val ? (float)stat_val->valueint : 0.0f;
        stat_val = cJSON_GetObjectItem(move_details, "priority");
        *feature_ptr++ = stat_val ? (float)stat_val->valueint : 0.0f;
    }
    return feature_ptr;
}

// Funzione principale che converte una linea JSON in un feature vector
void create_feature_vector(cJSON* root, float* feature_vector) {
    float* feature_ptr = feature_vector;

    // P1 Team Details (6 Pokemon)
    cJSON* p1_team = cJSON_GetObjectItem(root, "p1_team_details");
    for (int i = 0; i < 6; ++i) {
        feature_ptr = process_pokemon_details(cJSON_GetArrayItem(p1_team, i), feature_ptr);
    }

    // P2 Lead Details
    feature_ptr = process_pokemon_details(cJSON_GetObjectItem(root, "p2_lead_details"), feature_ptr);

    // Battle Timeline (fino a 30 turni, con padding)
    cJSON* timeline = cJSON_GetObjectItem(root, "battle_timeline");
    int num_turns = cJSON_GetArraySize(timeline);
    for (int i = 0; i < 30; ++i) {
        cJSON* turn_data = (i < num_turns) ? cJSON_GetArrayItem(timeline, i) : NULL;
        if (turn_data) {
            feature_ptr = process_player_turn_state(turn_data, "p1", feature_ptr);
            feature_ptr = process_player_turn_state(turn_data, "p2", feature_ptr);
        } else {
            // Padding con zeri se la battaglia ha meno di 30 turni
            memset(feature_ptr, 0, (PLAYER_TURN_STATE_SIZE * 2) * sizeof(float));
            feature_ptr += (PLAYER_TURN_STATE_SIZE * 2);
        }
    }

    // Standardizzazione finale (ottimizzata con AVX2)
    standardize_vector_avx(feature_vector, TOTAL_FEATURES);
}


// =============================================================================
// 4. LOGICA DI PARALLELIZZAZIONE E GESTIONE FILE (MODIFICATA PER WINDOWS)
// =============================================================================

typedef struct {
    char** lines;
    int start_index;
    int num_lines;
    float* features_buffer;
    float* labels_buffer;
} worker_args_t;

// =============================================================================
// MODIFICA 2: Cambiare la firma della funzione del thread per l'API Win32
// La funzione deve restituire DWORD e accettare LPVOID come argomento.
// =============================================================================
DWORD WINAPI worker_thread_func(LPVOID args) {
    worker_args_t* w_args = (worker_args_t*)args;
    for (int i = 0; i < w_args->num_lines; ++i) {
        int line_idx = w_args->start_index + i;
        cJSON* root = cJSON_Parse(w_args->lines[line_idx]);
        if (root) {
            // Estrai feature
            create_feature_vector(root, w_args->features_buffer + (long long)i * TOTAL_FEATURES);
            // Estrai label
            cJSON* won_node = cJSON_GetObjectItem(root, "player_won");
            w_args->labels_buffer[i] = (cJSON_IsTrue(won_node)) ? 1.0f : 0.0f;
            cJSON_Delete(root);
        }
    }
    return 0; // Restituisce 0 in caso di successo
}

// Funzione che processa un intero file .jsonl
void process_file(const char* file_path) {
    printf("Processing file: %s\n", file_path);

    // Costruisci i percorsi per i file di cache
    char features_path[256], labels_path[256];
    char temp_path[256];
    strcpy(temp_path, file_path);
    char* filename = strrchr(temp_path, '/');
    if (!filename) filename = strrchr(temp_path, '\\');
    filename = filename ? filename + 1 : temp_path;
    char* ext = strrchr(filename, '.');
    if (ext) *ext = '\0';
    
    sprintf(features_path, "./fds-challenge-dataset/kfolds/processed_datasets/%s_features.bin", filename);
    sprintf(labels_path, "./fds-challenge-dataset/kfolds/processed_datasets/%s_labels.bin", filename);

    // Controlla se la cache esiste
    FILE* f_cache = fopen(features_path, "rb");
    if (f_cache) {
        fclose(f_cache);
        printf("  -> Found pre-processed cache. Skipping.\n");
        return;
    }

    // Leggi tutte le righe del file in memoria
    FILE* f_in = fopen(file_path, "r");
    if (!f_in) {
        printf("  -> ERROR: Could not open file.\n");
        return;
    }

    char** lines = NULL;
    char buffer[1024 * 32]; // Buffer grande per righe JSON lunghe
    int line_count = 0;
    while (fgets(buffer, sizeof(buffer), f_in)) {
        lines = (char**)realloc(lines, (line_count + 1) * sizeof(char*));
        lines[line_count++] = _strdup(buffer);
    }
    fclose(f_in);
    printf("  -> Read %d lines from file.\n", line_count);

    // Alloca memoria per i risultati
    float* all_features = (float*)malloc((long long)line_count * TOTAL_FEATURES * sizeof(float));
    float* all_labels = (float*)malloc((long long)line_count * sizeof(float));
    if (!all_features || !all_labels) {
        printf("  -> ERROR: Memory allocation failed.\n");
        exit(1);
    }

    // =============================================================================
    // MODIFICA 3: Usare l'API Win32 per creare e attendere i thread
    // =============================================================================
    #define NUM_THREADS 8 // Per i7-7700K
    HANDLE threads[NUM_THREADS]; // Array di HANDLE per i thread
    worker_args_t args[NUM_THREADS];
    int lines_per_thread = line_count / NUM_THREADS;
    int remainder = line_count % NUM_THREADS;

    printf("  -> Starting processing with %d threads...\n", NUM_THREADS);
    clock_t start_time = clock();

    int current_line = 0;
    for (int i = 0; i < NUM_THREADS; ++i) {
        args[i].lines = lines;
        args[i].start_index = current_line;
        args[i].num_lines = lines_per_thread + (i < remainder ? 1 : 0);
        args[i].features_buffer = all_features + (long long)current_line * TOTAL_FEATURES;
        args[i].labels_buffer = all_labels + current_line;
        
        // Crea il thread usando CreateThread
        threads[i] = CreateThread(NULL, 0, worker_thread_func, &args[i], 0, NULL);
        current_line += args[i].num_lines;
    }

    // Attendi la fine di tutti i thread usando WaitForMultipleObjects
    WaitForMultipleObjects(NUM_THREADS, threads, TRUE, INFINITE);

    // Chiudi gli handle dei thread
    for (int i = 0; i < NUM_THREADS; ++i) {
        CloseHandle(threads[i]);
    }

    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("  -> Processing finished in %.2f seconds.\n", time_spent);

    // Salva i risultati nella cache
    FILE* f_feat_out = fopen(features_path, "wb");
    fwrite(all_features, sizeof(float), (long long)line_count * TOTAL_FEATURES, f_feat_out);
    fclose(f_feat_out);

    FILE* f_lab_out = fopen(labels_path, "wb");
    fwrite(all_labels, sizeof(float), line_count, f_lab_out);
    fclose(f_lab_out);
    printf("  -> Saved processed data to %s and %s\n", features_path, labels_path);

    // Libera memoria
    for (int i = 0; i < line_count; ++i) free(lines[i]);
    free(lines);
    free(all_features);
    free(all_labels);
}

// =============================================================================
// 5. FUNZIONE MAIN
// =============================================================================

int main() {
    // Carica le mappe una sola volta all'inizio
    printf("Loading data mappings...\n");
    load_map_from_json("./fds-challenge-dataset/pokedex.json", &pokemon_map);
    load_map_from_json("./fds-challenge-dataset/moves.json", &move_map);
    load_map_from_json("./fds-challenge-dataset/statuses.json", &status_map);
    load_map_from_json("./fds-challenge-dataset/types.json", &type_map);
    // Aggiungi manualmente 'nostatus' se necessario, come nello script Python
    if (get_index_from_map(status_map, "nostatus", false) == -1) {
        mapping_t* item = (mapping_t*)malloc(sizeof(mapping_t));
        item->name = _strdup("nostatus");
        item->index = HASH_COUNT(status_map);
        HASH_ADD_KEYPTR(hh, status_map, item->name, strlen(item->name), item);
    }
    printf("Mappings loaded.\n\n");

    // Crea la directory per i dati processati se non esiste
    _mkdir("./fds-challenge-dataset/kfolds/processed_datasets");

    // Ciclo principale sui fold, come nello script Python
    const int NUM_FOLDS = 10;
    for (int fold = 1; fold <= NUM_FOLDS; ++fold) {
        printf("===== FOLD %d/%d =====\n", fold, NUM_FOLDS);
        char train_file[256], val_file[256];
        sprintf(train_file, "./fds-challenge-dataset/kfolds/fold_%d_train.jsonl", fold);
        sprintf(val_file, "./fds-challenge-dataset/kfolds/fold_%d_val.jsonl", fold);

        process_file(train_file);
        process_file(val_file);
        printf("\n");
    }

    printf("All folds processed.\n");

    // TODO: Liberare la memoria delle hash map (non strettamente necessario alla fine del main)
    return 0;
}