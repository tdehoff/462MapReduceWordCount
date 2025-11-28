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

vector<string> readers_q;
vector<vector<pair<string, int>>> send_buffers;
vector<vector<pair<string, int>>> reducer_queues;
unordered_map<string, int> global_counts;

size_t total_words;
size_t files_remain;
int num_reducers;
int num_readers;
int readers_avail;
int total_ranks;
int num_mappers;

omp_lock_t readers_lock;
vector<omp_lock_t> mappers_locks;
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
    for (char &ch : w) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (isupper(c)) {
            ch = tolower(c);
        }
    }
}

void read_file (char* fname) {
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
    readers_q.insert(readers_q.end(), make_move_iterator(words.begin()), make_move_iterator(words.end()));
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

void mapping_step() {
    unordered_map<string, int> buckets;

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
            // Mappers are ahead of readers
            #pragma omp taskyield
        }
    }

    // Push thread's results into the reducer queues
    for (auto el : buckets) {
        int dst_rank = hash_str(el.first, total_ranks);

        omp_set_lock(&mappers_locks[dst_rank]);
        send_buffers[dst_rank].push_back(el);
        omp_unset_lock(&mappers_locks[dst_rank]);
    }
}

void exchange_data(int my_rank) {
    for (int i = 0; i < total_ranks; ++i) {
        // Skip sending to yourself, send to reducer queues
        if (i == my_rank) {
            for (auto &el : send_buffers[i]) {
                int ind = hash_str(el.first, num_reducers);
                reducer_queues[ind].push_back(el);
            }
        }
        else {
            // Send total number of elements first
            int send_N = send_buffers[i].size();
            int rec_N;

            MPI_Sendrecv(&send_N, 1, MPI_INT, i, 0,
                         &rec_N, 1, MPI_INT, i, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int j = 0; j < max(send_N, rec_N); ++j) {
                int send_len = 0;
                int send_count = 0;
                string *send_word = nullptr;

                if (j < send_N) {
                    send_word = &send_buffers[i][j].first;
                    send_len = send_word->length();
                    send_count = send_buffers[i][j].second;
                }

                int recv_len = 0;
                int recv_count = 0;
                string recv_word;

                // Exchange lengths
                MPI_Sendrecv(
                    &send_len, 1, MPI_INT, i, 0,
                    &recv_len, 1, MPI_INT, i, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                );
                recv_word.resize(recv_len);

                // Exchange words
                MPI_Sendrecv(
                    send_len ? send_word->data() : nullptr, send_len, MPI_CHAR, i, 0,
                    recv_len ? &recv_word[0] : nullptr, recv_len, MPI_CHAR, i, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                );

                // Exchange counts
                MPI_Sendrecv(
                    &send_count, 1, MPI_INT, i, 0,
                    &recv_count, 1, MPI_INT, i, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                );

                // Store received element
                if (recv_len > 0) {
                    int idx = hash_str(recv_word, num_reducers);
                    reducer_queues[idx].push_back({recv_word, recv_count});
                }
            }
        }
    }
}

void reduce_step(int id) {
    // Use local hash table for partial results
    unordered_map<string, int> local_result;
    for (auto &cur_entry : reducer_queues[id]) {
        local_result[cur_entry.first] += cur_entry.second;
    }
    // Merge partial results into global results
    omp_set_lock(&global_counts_lock);
    for (auto &el : local_result) {
        global_counts[el.first] += el.second;
    }
    omp_unset_lock(&global_counts_lock);
}

void gather_results(int my_rank) {
    if (my_rank == 0) {
        for (int  i = 1; i < total_ranks; ++i) {
            int N;
            MPI_Recv(&N, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int j = 0; j < N; ++j) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                string w;
                w.resize(len);
                MPI_Recv(&w[0], len, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int count;
                MPI_Recv(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                global_counts[w] += count;
            }
        }
    }
    else {
        int N = global_counts.size();
        MPI_Send(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        for (auto &el : global_counts) {
            string w = el.first;
            int len = w.length();
            int count = el.second;

            MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(w.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
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
    }

    if (argc < 2) {
        fprintf(stderr, "usage: %s <input_files>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n_threads = omp_get_max_threads();
    num_reducers = n_threads * 2;  // Works best on my laptop -- test on ISAAC
    files_remain = 0;

    num_readers = n_threads;
    num_mappers = n_threads;
    readers_avail = num_readers;
    total_ranks = size;

    if (rank == 0) {
        cerr << "Testing " <<  n_threads << " thread(s), " << size << " processes\n";
    }

    omp_init_lock(&readers_lock);
    omp_init_lock(&global_counts_lock);
    reducer_locks.resize(num_reducers);
    for (int i = 0; i < num_reducers; ++i) {
        omp_init_lock(&reducer_locks[i]);
    }
    mappers_locks.resize(total_ranks);
    for (int i = 0; i < total_ranks; ++i) {
        omp_init_lock(&mappers_locks[i]);
    }
    reducer_queues.resize(num_reducers);
    send_buffers.resize(total_ranks);

    double start, end, start_c, start_r, start_p;
    start = MPI_Wtime();

    #pragma omp parallel
    {
        #pragma omp master
        {
            // File reading step
            if (rank == 0) {
                int f_count = 1;
                size_t active_ranks = size - 1;
                bool done = false;
                MPI_Status stat;
                int tmp;
                int flag;
                int local_avail;

                while (active_ranks > 0 || !done) {
                    // Check if any ranks sent a pending request
                    MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag, &stat);

                    // If not, generate tasks for master rank theads
                    #pragma omp atomic read
                    local_avail = readers_avail;

                    if (!done && !flag && local_avail > 0) {
                        if (f_count < argc) {
                            #pragma omp atomic
                            files_remain++;

                            #pragma omp task
                            {
                                read_file(argv[f_count]);
                            }
                            f_count++;
                        }
                        else {
                            done = true;
                        }
                    }
                    else if (size > 1) {
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

                        #pragma omp atomic
                        files_remain++;

                        #pragma omp task
                        {
                            read_file(argv[rec_buff]);
                        }
                    }
                }
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
    start_c = MPI_Wtime();

    exchange_data(rank);

    start_r = MPI_Wtime();
    // Reducing step
    #pragma omp parallel for
    for (int i = 0; i < num_reducers; ++i) {
        reduce_step(i);
    }

    // Nothing to gather for single rank
    if (total_ranks > 1) {
        gather_results(rank);
    }

    size_t global_total_words = 0;
    MPI_Reduce(&total_words, &global_total_words, 1,
            MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        total_words = global_total_words;
    }

    start_p = MPI_Wtime();
    vector<pair<string, int>> counts;
    for (auto &el : global_counts) {
        counts.emplace_back(el.first, el.second);
    }

    // Sort in alphabetical order
    sort(counts.begin(), counts.end(),
         [](const auto &a, const auto &b) {
        return a.first < b.first;
    });

    // Print step
    if (rank == 0) {
        ofstream out("hybrid_out.txt");
        out << "Filename: " << argv[1] << ", total words: " << global_total_words << "\n";
        // ISAAC is having issues printing too much output, only print the number of unique words
        // Error: srun: error: eio_handle_mainloop: Abandoning IO 60 secs after job shutdown initiated
        out << "Unique words found: " << counts.size() << "\n";
        for (size_t i = 0; i < counts.size(); ++i) {
            out << "[" << i << "] " << counts[i].first << ": " << counts[i].second << "\n";
        };
    }

    end = MPI_Wtime();
    if (rank == 0) {
        // Use cerr to always print in terminal
        cerr << "Hybrid time: " << (end - start) * 1000 << " ms\n";
        cerr << "  File read & Map time: " << (start_c - start) * 1000 << " ms\n";
        cerr << "  Communication time: " << (start_r - start_c) * 1000 << " ms\n";
        cerr << "  Reducing time: " << (start_p - start_r) * 1000 << " ms\n";
        cerr << "  Sort & Print time: " << (end - start_p) * 1000 << " ms\n";
    }

    omp_destroy_lock(&readers_lock);
    omp_destroy_lock(&global_counts_lock);
    for (int i = 0; i < num_reducers; ++i) {
        omp_destroy_lock(&reducer_locks[i]);
    }

    MPI_Finalize();
    return 0;
}
