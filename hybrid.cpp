#include <fstream>
#include <unordered_map>
#include <string>
#include <cctype>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <queue>
#include <mpi.h>

using namespace std;

unordered_map<string, size_t> readers_map;
vector<queue<pair<string, size_t>>> reducer_queues;
unordered_map<string, size_t> global_counts;

size_t total_words;
size_t files_remain;
int num_reducers;
int num_readers;
int readers_avail;

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

void read_and_map (char* fname) {
    #pragma omp atomic
    readers_avail--;

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
    for (string &s : words) {
        readers_map[s]++;
    }
    omp_unset_lock(&readers_lock);

    #pragma omp atomic
    total_words += wc;

    #pragma omp atomic
    files_remain--;

    #pragma omp atomic
    readers_avail++;
}

int hash_str(string s, int R) {
    int sum = 0;
    for (unsigned char c : s) {
        sum  += c;
    }
    return sum % R;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        printf("Error: MPI_THREAD_FUNNELED is not supported.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
        printf("Rank %d: threading level provided = %d\n", rank, provided);
    }

    if (argc < 2) {
        fprintf(stderr, "usage: %s <input_files>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n_threads = omp_get_max_threads();
    num_reducers = n_threads * 2;  // Works best on my laptop -- test on ISAAC
    files_remain = argc - 1;

    num_readers = n_threads;
    readers_avail = num_readers;

    if (rank == 0) {
        cerr << "Testing " <<  n_threads << " thread(s), " << size << " processes\n";
    }

    omp_init_lock(&readers_lock);
    omp_init_lock(&global_counts_lock);
    reducer_locks.resize(num_reducers);
    for (int i = 0; i < num_reducers; ++i) {
        omp_init_lock(&reducer_locks[i]);
    }
    reducer_queues.resize(num_reducers);

    double start, end, start_r, start_p;
    start = MPI_Wtime();

    #pragma omp parallel
    {
        #pragma omp master
        {
            // File reading step
            if (rank == 0) {
                size_t f_count = 1;
                size_t active_ranks = size - 1;
                MPI_Status stat;
                int tmp;
                int flag;
                int local_avail;

                while (active_ranks > 0) {
                    // Check if any ranks sent a pending request
                    MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag, &stat);

                    // If not, generate tasks for master rank theads
                    #pragma omp atomic read
                    local_avail = readers_avail;
                    if (!flag && local_avail > 0) {
                        if (f_count < argc) {
                            #pragma omp task
                            {
                                read_and_map(argv[f_count]);
                            }
                            f_count++;
                        }
                    }
                    else {
                        // Use tag = 1 for requests
                        MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &stat);
                        int requesting_rank = stat.MPI_SOURCE;

                        int send_buff = -1;
                        if (f_count < argc) {
                            send_buff = f_count;
                            f_count++;
                        }
                        else {
                            // This rank receives -1 for "work done"
                            active_ranks--;
                        }
                        // Use tag = 2 for responds
                        MPI_Send(&send_buff, 1, MPI_INT, requesting_rank, 2, MPI_COMM_WORLD);
                    }
                }
            }
            else {
                int local_avail;
                int rec_buff = 0;
                while (true) {
                    #pragma omp atomic read
                    local_avail = readers_avail;
                    if (local_avail > 0) {
                        // Send request
                        MPI_Send(&rec_buff, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
                        // Receive file number or -1 for "work done"
                        MPI_Recv(&rec_buff, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        if (rec_buff == -1) {
                            break;
                        }

                        #pragma omp task
                        {
                            read_and_map(argv[rec_buff]);
                        }
                    }
                }
            }
        }
    }
    end = MPI_Wtime();
    cerr << "File reading + mapping took " << (end - start) * 1000 << " ms\n";

    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     {


    //         while (argv[f_count]) {
    //             #pragma omp task firstprivate(f_count)
    //             {
    //                 read_file(argv[f_count]);
    //             }
    //             f_count++;
    //         }

    //         // Mapping step
    //         for (int i = 0; i < num_mappers; ++i) {
    //             #pragma omp task
    //             {
    //                 mapping_step();
    //             }
    //         }
    //     }
    // }

    // start_r = omp_get_wtime();
    // // Reducing step
    // #pragma omp parallel for
    // for (int i = 0; i < num_reducers; ++i) {
    //     reduce_step(i);
    // }

    // start_p = omp_get_wtime();
    // vector<pair<string, size_t>> counts;
    // for (auto &el : global_counts) {
    //     counts.emplace_back(el.first, el.second);
    // }

    // // Sort in alphabetical order
    // sort(counts.begin(), counts.end(),
    //      [](const auto &a, const auto &b) {
    //     return a.first < b.first;
    // });

    // // Print step
    // cout << "Filename: " << argv[1] << ", total words: " << total_words << endl;
    // for (size_t i = 0; i < counts.size(); ++i) {
    //     cout << "[" << i << "] " << counts[i].first << ": " << counts[i].second << endl;
    // }

    // end = omp_get_wtime();
    // // Use cerr to always print in terminal
    // cerr << "OpenMP time: " << (end - start) * 1000 << " ms\n";
    // cerr << "  File read & Map time: " << (start_r - start) * 1000 << " ms\n";
    // cerr << "  Reducing time: " << (start_p - start_r) * 1000 << " ms\n";
    // cerr << "  Sort & Print time: " << (end - start_p) * 1000 << " ms\n";

    // omp_destroy_lock(&readers_lock);
    // omp_destroy_lock(&global_counts_lock);
    // for (int i = 0; i < num_reducers; ++i) {
    //     omp_destroy_lock(&reducer_locks[i]);
    // }

    MPI_Finalize();
    return 0;
}
