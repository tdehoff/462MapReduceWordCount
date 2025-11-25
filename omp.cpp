#include <fstream>
#include <unordered_map>
#include <string>
#include <cctype>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <queue>

using namespace std;

queue<string> readers_q;
vector<queue<pair<string, size_t>>> reducer_queues;
unordered_map<string, size_t> global_counts;

size_t total_words;
size_t files_remain;
int num_reducers;

omp_lock_t readers_lock;
vector<omp_lock_t> reducer_locks;
omp_lock_t global_counts_lock;

void process_word(string &w) {
    // Remove punctuation at beginning
    while (!w.empty() && ispunct(w[0])) {
        w.erase(0, 1);
    }
    // Remove punctuation at end
    while (!w.empty() && ispunct(w[w.size() - 1])) {
        w.pop_back();
    }
    // Convert all letters to lowercase
    for (size_t i = 0; i < w.length(); ++i) {
        if (isupper(w[i])) {
            w[i] = tolower(w[i]);
        }
    }
}

void read_file (char* fname) {
    size_t wc = 0;
    ifstream fin(fname);
    if (!fin) {
        fprintf(stderr, "error: unable to open input file: %s\n", fname);
        exit(1);
    }

    string word;
    while (fin >> word) {
        process_word(word);
        if (!word.empty()) {          // avoid pushing empty strings
            wc++;
            omp_set_lock(&readers_lock);
            readers_q.push(word);
            omp_unset_lock(&readers_lock);
        }
    }
    #pragma omp atomic
    total_words += wc;

    #pragma omp atomic
    files_remain--;
}

int hash_str(string s, int R) {
    int sum = 0;
    for (unsigned char c : s) {
        sum  += c;
    }
    return sum % R;
}

void mapping_step() {
    unordered_map<string, size_t> buckets;

    // Grab elemnts from the work q in chunks
    const int chunk_size = 1024;  // find which chunk size works the best
    vector<string> working_batch(chunk_size);

    while (true) {
        working_batch.clear();

        // Lock and grab new chunk of elements if queue is not empty
        omp_set_lock(&readers_lock);
        for (size_t i = 0; i < chunk_size && !readers_q.empty(); ++i) {
            working_batch.push_back(readers_q.front());
            readers_q.pop();
        }
        omp_unset_lock(&readers_lock);

        if (!working_batch.empty()) {
            // Queue not empty -- process new elements
            for (size_t i = 0; i < working_batch.size(); ++i) {
                buckets[working_batch[i]]++;
            }
        }
        else {
            int remaining;
            // Shared global variable -- must be read atomically
            #pragma omp atomic read
            remaining = files_remain;

            if (remaining == 0) {
                // Queue empty and all files are processed
                break;
            }
            else {
                // Mappers are ahead of readers
                #pragma omp taskyield
            }
        }
    }

    // Push thread's results into the reducer queues
    for (auto el : buckets) {
        int index = hash_str(el.first, num_reducers);
        omp_set_lock(&reducer_locks[index]);
        reducer_queues[index].push(el);
        omp_unset_lock(&reducer_locks[index]);
    }
}

void reduce_step(int id) {
    // Use local hash table for partial results
    unordered_map<string, size_t> local_result;
    while (!reducer_queues[id].empty()) {
        pair<string, size_t> cur_entry = reducer_queues[id].front();
        reducer_queues[id].pop();
        local_result[cur_entry.first] += cur_entry.second;
    }
    // Merge partial results into global results
    omp_set_lock(&global_counts_lock);
    for (auto &el : local_result) {
        global_counts[el.first] += el.second;
    }
    omp_unset_lock(&global_counts_lock);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input_files>\n", argv[0]);
        return 1;
    }

    int n_threads = 4;
    omp_set_num_threads(n_threads);

    int num_mappers = n_threads / 2;
    num_reducers = n_threads;
    files_remain = argc - 1;

    omp_init_lock(&readers_lock);
    omp_init_lock(&global_counts_lock);
    reducer_locks.resize(num_reducers);
    for (int i = 0; i < num_reducers; ++i) {
        omp_init_lock(&reducer_locks[i]);
    }
    reducer_queues.resize(num_reducers);

    double start, end, start_f, end_f, start_m, end_m, start_r, end_r, start_p;
    start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            start_f = omp_get_wtime();
            // File reading step
            size_t f_count = 1;
            while (argv[f_count]) {
                #pragma omp task firstprivate(f_count)
                {
                    read_file(argv[f_count]);
                }
                f_count++;
            }
            end_f = omp_get_wtime();

            start_m = omp_get_wtime();
            // Mapping step
            for (int i = 0; i < num_mappers; ++i) {
                #pragma omp task
                {
                    mapping_step();
                }
            }

            // Wait for readers + reducers to complete
            #pragma omp taskwait
            end_m = omp_get_wtime();

            start_r = omp_get_wtime();
            // Reducing step
            for (int i = 0; i < num_reducers; ++i) {
                #pragma omp task firstprivate(i)
                {
                    reduce_step(i);
                }
            }
            end_r = omp_get_wtime();
        }
    }

    start_p = omp_get_wtime();
    vector<pair<string, size_t>> counts;
    for (auto &el : global_counts) {
        counts.emplace_back(el.first, el.second);
    }

    // Sort in alphabetical order
    sort(counts.begin(), counts.end(),
         [](const auto &a, const auto &b) {
        return a.first < b.first;
    });

    // Print step
    cout << "Filename: " << argv[1] << ", total words: " << total_words << endl;
    for (size_t i = 0; i < counts.size(); ++i) {
        cout << "[" << i << "] " << counts[i].first << ": " << counts[i].second << endl;
    }

    end = omp_get_wtime();
    // Use cerr to always print in terminal
    cerr << "OpenMP time: " << (end - start) * 1000 << " ms\n";
    cerr << "  File reading time: " << (end_f - start_f) * 1000 << " ms\n";
    cerr << "  Mapping time: " << (end_m - start_m) * 1000 << " ms\n";
    cerr << "  Reducing time: " << (end_r - start_r) * 1000 << " ms\n";
    cerr << "  Sort & Print time: " << (end - start_p) * 1000 << " ms\n";

    omp_destroy_lock(&readers_lock);
    omp_destroy_lock(&global_counts_lock);
    for (int i = 0; i < num_reducers; ++i) {
        omp_destroy_lock(&reducer_locks[i]);
    }

    return 0;
}
