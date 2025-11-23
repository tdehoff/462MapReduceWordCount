#include <cstdio>
#include <fstream>
#include <unordered_map>
#include <string>
#include <cctype>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void process_word(string &w) {
    // cout << "DEBUG: " << w << " ---> ";           // DEBUG PRINT
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
    // cout << w << endl;                              // DEBUG PRINT
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    ifstream fin(argv[1]);
    if (!fin) {
        fprintf(stderr, "error: unable to open input file: %s\n", argv[1]);
        return 1;
    }

    string word;
    vector<pair<string, size_t>> raw_tuples;

    while (fin >> word) {
        process_word(word);
        // Map step
        raw_tuples.push_back(pair(word, 1));
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

    // Sort for print consistency: larger counts first
    // May remove after we verify correctness
    sort(counts.begin(), counts.end(), [](const pair<string, int> &a, const pair<string, int> &b) {
        return a.second > b.second;
    });

    // Print step
    for (size_t i = 0; i < counts.size(); ++i) {
        cout << "[" << i << "] " << counts[i].first << ": " << counts[i].second << endl;
    }

    return 0;
}