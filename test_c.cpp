#include <iostream>
#include <random>
#include <chrono>

using namespace std;

int main() {
    // 使用随机设备作为种子
    random_device rd;
    mt19937 gen(rd()); // Mersenne Twister 随机数引擎
    
    // 生成 [0.0, 1.0) 之间的均匀分布实数
    uniform_real_distribution<double> dis(0.0, 1.0);
    
    int N=1000;
    double x[N], y[N], z[N], m[N];

    for (int i = 0; i < N; ++i) {
        x[i] = dis(gen);
        y[i] = dis(gen);
        z[i] = dis(gen);
        m[i] = dis(gen);
    }
    auto start = chrono::high_resolution_clock::now();
    double ans = 0;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (i==j) break;
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            double mm = m[i] * m[j];
            double d = sqrt(dx * dx + dy * dy + dz * dz);
            ans += mm * dx / (d*d*d);
            ans += mm * dy / (d*d*d);
            ans += mm * dz / (d*d*d);
        }
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;
    return 0;
}