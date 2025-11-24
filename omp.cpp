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

size_t read_file (char* fname, queue<pair<string, size_t>> &q) {
    size_t wc = 0;
    ifstream fin(fname);
    if (!fin) {
        fprintf(stderr, "error: unable to open input file: %s\n", fname);
        exit(1);
    }

    string word;
    while (fin >> word) {
        wc++;
        process_word(word);
        if (!word.empty()) {          // avoid pushing empty strings
            # pragma omp critical
            {
                q.push(pair(word, 1));
            }
        }
    }
    return wc;
}

int hash(string s, int R) {
    int sum = 0;
    for (unsigned char c : s) {
        sum  += c;
    }
    return sum % R;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input_files>\n", argv[0]);
        return 1;
    }

    int n_threads = 4;
    omp_set_num_threads(n_threads);
    double start, end;
    start = omp_get_wtime();

    queue<pair<string, size_t>> readers_q;
    vector<queue<pair<string, size_t>>> reducer_queues(n_threads);
    size_t total_word_count = 0;
    size_t files_remain = argc - 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            // File reading step
            size_t f_count = 1;
            while (argv[f_count]) {
                #pragma omp task firstprivate(f_count)
                {
                    size_t wc = read_file(argv[f_count], readers_q);

                    #pragma omp atomic
                    total_word_count += wc;

                    #pragma omp atomic
                    files_remain--;
                }
                f_count++;
            }

            // Mapping step
            // int num_mappers = n_threads / 2;
            for (int i = 0; i < n_threads; ++i) {
                #pragma omp task
                {
                    // NOTE: may want to isolate this code in a map() function
                    unordered_map<string, size_t> combined_recs;
                    while (true) {
                        bool not_empty = 0;
                        pair<string, size_t> cur_element;
                        #pragma omp critical
                        {
                            // Lock and grab new element if queue is not empty
                            if (not_empty = !readers_q.empty()) {
                                cur_element = readers_q.front();
                                readers_q.pop();
                            }
                        }
                        if (not_empty) {
                            // Queue not empty -- process new element
                            combined_recs[cur_element.first]++;
                        }
                        else if (files_remain == 0) {
                            // Queue empty and all files are processed
                            break;
                        }
                    }
                    // TODO: push map elements to reducer's queue, compute index using hash(first, num_reducers)
                }
            }

        }


        // Shuffle step
        unordered_map<string, vector<size_t>> buckets;
        for (size_t i = 0; i < raw_tuples.size(); ++i) {
            buckets[raw_tuples[i].first].push_back(raw_tuples[i].second);
        }

        // Reduce step
        vector<pair<string, size_t>> counts;
        for (auto entry : buckets) {
            size_t sum = 0;
            for (size_t i = 0; i < entry.second.size(); ++i) {
                sum += entry.second[i];
            }
            counts.push_back(pair(entry.first, sum));
        }

    }

    // Sort in alphabetical order
    sort(counts.begin(), counts.end(), [](const pair<string, int> &a, const pair<string, int> &b) {
        return a.first < b.first;
    });

    // Print step
    cout << "Filename: " << argv[1] << ", total words: " << total_word_count << endl;
    for (size_t i = 0; i < counts.size(); ++i) {
        cout << "[" << i << "] " << counts[i].first << ": " << counts[i].second << endl;
    }

    end = omp_get_wtime();
    // Use cerr to always print in terminal
    cerr << "Sequential time: " << (end - start) * 1000 << " ms\n";

    return 0;
}
