#include <iostream>
#include <vector>
using namespace std;

const int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
vector<vector<int>> tipPoints;

void find_zero_with_one_neighbor(vector<vector<int>>& grid) {
    int rows = grid.size();
    int cols = grid[0].size();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Check if the current point is 0
            if (grid[i][j] == 0) {
                int neighborCount = 0;

                // Check all 8 possible neighbors
                for (auto& dir : directions) {
                    int nx = i + dir[0];
                    int ny = j + dir[1];

                    // If the neighbor is within bounds and is 0
                    if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && grid[nx][ny] == 0) {
                        neighborCount++;
                    }
                }

                // If there's exactly one neighboring 0, add it to tipPoints
                if (neighborCount == 1) {
                    tipPoints.push_back({i, j});
                }
            }
        }
    }
}

int main() {
    vector<vector<int>> grid = {
        {1, 1, 0, 1, 1},
        {1, 0, 0, 1, 1},
        {1, 0, 1, 1, 1},
        {1, 0, 1, 1, 1}
    };

    find_zero_with_one_neighbor(grid);

    cout << "Points with exactly one neighboring 0:" << endl;
    for (const auto& point : tipPoints) {
        cout << "(" << point[0] << ", " << point[1] << ")" << endl;
    }

    return 0;
}
