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

vector<string> readers_q;
vector<queue<pair<string, size_t>>> reducer_queues;
unordered_map<string, size_t> global_counts;

size_t total_words;
size_t files_remain;
int num_reducers;

omp_lock_t readers_lock;
vector<omp_lock_t> reducer_locks;
omp_lock_t global_counts_lock;

void process_word(string &w) {
    // Remove punctuation and non-ascii chars at beginning
    while (!w.empty()) {
        signed char c = w.front();
        if (c < 0 || ispunct(c)) {
            w.erase(0, 1);
            continue;
        }
        break;
    }
    // Remove punctuation and non-ascii chars at end
    while (!w.empty()) {
        signed char c = w.back();
        if (c < 0 || ispunct(c)) {
            w.pop_back();
            continue;
        }
        break;
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

    // Process words in chunks to reduce locking
    const int chunk_size = 1024;  // select the best chunk size
    vector<string> words;
    words.reserve(chunk_size);

    string word;
    while (fin >> word) {
        process_word(word);
        if (!word.empty()) {          // avoid pushing empty strings
            wc++;
            words.push_back(word);
        }
    }
    omp_set_lock(&readers_lock);
    readers_q.insert(readers_q.end(), make_move_iterator(words.begin()), make_move_iterator(words.end()));
    omp_unset_lock(&readers_lock);

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
    vector<string> working_batch;
    working_batch.reserve(chunk_size);

    while (true) {
        working_batch.clear();

        // Lock and grab new chunk of elements if queue is not empty
        omp_set_lock(&readers_lock);
        for (size_t i = 0; i < chunk_size && !readers_q.empty(); ++i) {
            working_batch.push_back(readers_q.back());
            readers_q.pop_back();
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

    int n_threads = omp_get_max_threads();
    int num_mappers = n_threads;
    num_reducers = n_threads * 2;  // Works best on my laptop -- test on ISAAC
    files_remain = argc - 1;\

    cerr << "Testing " <<  n_threads << " thread(s)\n";

    omp_init_lock(&readers_lock);
    omp_init_lock(&global_counts_lock);
    reducer_locks.resize(num_reducers);
    for (int i = 0; i < num_reducers; ++i) {
        omp_init_lock(&reducer_locks[i]);
    }
    reducer_queues.resize(num_reducers);

    double start, end, start_r, start_p;
    start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            // File reading step
            size_t f_count = 1;
            while (argv[f_count]) {
                #pragma omp task firstprivate(f_count)
                {
                    read_file(argv[f_count]);
                }
                f_count++;
            }

            // Mapping step
            for (int i = 0; i < num_mappers; ++i) {
                #pragma omp task
                {
                    mapping_step();
                }
            }
        }
    }

    start_r = omp_get_wtime();
    // Reducing step
    #pragma omp parallel for
    for (int i = 0; i < num_reducers; ++i) {
        reduce_step(i);
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
    cerr << "  File read & Map time: " << (start_r - start) * 1000 << " ms\n";
    cerr << "  Reducing time: " << (start_p - start_r) * 1000 << " ms\n";
    cerr << "  Sort & Print time: " << (end - start_p) * 1000 << " ms\n";

    omp_destroy_lock(&readers_lock);
    omp_destroy_lock(&global_counts_lock);
    for (int i = 0; i < num_reducers; ++i) {
        omp_destroy_lock(&reducer_locks[i]);
    }

    return 0;
}
