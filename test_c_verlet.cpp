#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>

// 三维向量结构体
struct Vec3 {
    double x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    Vec3 operator*(double scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    
    double length() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

// 质点类
class Particle {
public:
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    Vec3 prev_acceleration;
    double mass;
    
    Particle(const Vec3& pos, const Vec3& vel, double m) 
        : position(pos), velocity(vel), mass(m) {
        acceleration = Vec3();
        prev_acceleration = Vec3();
    }
};

// 物理系统类
class NBodySystem {
private:
    std::vector<Particle> particles;
    double G;  // 万有引力常数
    double softening;  // 软化长度，防止数值不稳定
    
public:
    NBodySystem(double gravity, double softening) 
        : G(gravity), softening(softening) {}
    
    // 添加质点
    void addParticle(const Particle& p) {
        particles.push_back(p);
    }
    
    // 计算所有质点间的引力
    void computeForces() {
        const int n = particles.size();
        
        // 重置加速度
        for (auto& p : particles) {
            p.prev_acceleration = p.acceleration;
            p.acceleration = Vec3();
        }
        
        // 计算每对质点间的引力
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                Vec3 delta = particles[j].position - particles[i].position;
                double dist = delta.length();
                double dist_cubed = (dist*dist + softening*softening) * std::sqrt(dist*dist + softening*softening);
                double force_magnitude = G * particles[i].mass * particles[j].mass / dist_cubed;
                
                Vec3 force = delta * force_magnitude;
                
                particles[i].acceleration = particles[i].acceleration + force * (1.0 / particles[i].mass);
                particles[j].acceleration = particles[j].acceleration - force * (1.0 / particles[j].mass);
            }
        }
    }
    
    // Verlet 积分步
    void verletStep(double dt) {
        for (auto& p : particles) {
            Vec3 temp_pos = p.position;
            
            // 更新位置: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            p.position = p.position + p.velocity * dt + p.acceleration * (0.5 * dt * dt);
            
            // 更新速度: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
            p.velocity = p.velocity + (p.acceleration + p.prev_acceleration) * (0.5 * dt);
        }
    }
    
    // 模拟一步
    void simulateStep(double dt) {
        computeForces();
        verletStep(dt);
    }
    
    // 获取质点数量
    size_t size() const { return particles.size(); }
    
    // 获取质点位置
    const Vec3& getPosition(int i) const { return particles[i].position; }
};

// 生成随机数
double randomDouble(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

int main() {
    const int N = 1000;  // 质点数量
    const double G = 6.674e-11;  // 万有引力常数
    const double softening = 1.0;  // 软化长度
    const double dt = 0.1;  // 时间步长
    const int steps = 500;  // 模拟步数
    const double box_size = 100.0;  // 初始分布区域大小
    
    NBodySystem system(G, softening);
    
    // 初始化随机质点
    for (int i = 0; i < N; ++i) {
        Vec3 pos(randomDouble(-box_size, box_size), 
                randomDouble(-box_size, box_size), 
                randomDouble(-box_size, box_size));
        Vec3 vel(randomDouble(-1, 1), 
                randomDouble(-1, 1), 
                randomDouble(-1, 1));
        double mass = randomDouble(1.0, 10.0);
        
        system.addParticle(Particle(pos, vel, mass));
    }
    
    // 打开输出文件
    std::ofstream outfile("trajectories.txt");
    
    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 模拟循环
    for (int step = 0; step < steps; ++step) {
        system.simulateStep(dt);
        
        // 输出当前位置到文件
        for (int i = 0; i < system.size(); ++i) {
            const Vec3& pos = system.getPosition(i);
            outfile << pos.x << " " << pos.y << " " << pos.z << " ";
        }
        outfile << "\n";
        
        // 打印进度
        if (step % 10 == 0) {
            std::cout << "Step " << step << "/" << steps << " completed\n";
        }
    }
    
    // 结束计时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Simulation completed in " << duration.count() << " ms\n";
    
    outfile.close();
    
    std::cout << "Trajectories saved to trajectories.txt\n";
    
    return 0;
}