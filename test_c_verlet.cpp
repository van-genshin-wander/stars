#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>
#include <math.h>

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
    double radius;
    
    Particle(const Vec3& pos, const Vec3& vel, double m, double r) 
        : position(pos), velocity(vel), mass(m), radius(r) {
        acceleration = Vec3();
        prev_acceleration = Vec3();
    }
};

Particle sample(double R, double r, double G, double M, double mass, double radi){
    // 随机数生成器
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    // 1. 随机选择圆环上的角度（大圆）
    double theta = 2 * M_PI * dist(gen);
    
    // 2. 随机选择小圆上的角度
    double phi = 2 * M_PI * dist(gen);
    
    // 3. 随机选择小圆内的半径（考虑面积均匀分布）
    double radius = r * sqrt(dist(gen));
    
    // 4. 计算3D坐标
    Vec3 p, v;
    p.x = (R + radius * cos(phi)) * cos(theta);
    p.y = (R + radius * cos(phi)) * sin(theta);
    p.z = radius * sin(phi);
    double v_norm = std::sqrt(G * M / R);
    v.x = - v_norm * (cos(theta) + 0.01 * dist(gen));
    v.y = v_norm * (sin(theta) + 0.01 * dist(gen));
    v.z = v_norm * 0.02 * (dist(gen) - 0.5);

    Particle ans(p, v, mass, radi);

    return ans;
}


// 物理系统类
class NBodySystem {
private:
    std::vector<Particle> particles;
    double G;  // 万有引力常数
    double M;
    double softening;  // 软化长度，防止数值不稳定

public:
    int index = 0; //记录还存在的粒子个数
    NBodySystem(double gravity, double M, double softening) 
        : G(gravity), M(M), softening(softening) {}
    
    // 添加质点
    void addParticle(const Particle& p) {
        particles.push_back(p);
        index++;
    }
    
    void merge_particle(int i, int j) {
        auto& p1 = particles[i]; 
        auto& p2 = particles[j];
        double m1 = p1.mass; double m2 = p2.mass;
        double r1 = m1 / (m1 + m2); double r2 = m2 / (m1 + m2);
        p1.position = p1.position * r1 + p2.position * r2;
        p1.velocity = p1.velocity * r1 + p2.velocity * r2;
        p1.radius = pow(pow(p1.radius, 3) + pow(p2.radius, 3), 1.0/3);
        p1.mass = m1 + m2;
        p2 = particles[index - 1];
        index -= 1;
    }

    void check_collapse(){
        int flag = 0;
        for (int i = 0; i < index; i++){
            for (int j = i + 1; j < index; j++){
                auto p1 = particles[i];
                auto p2 = particles[j];
                if ((p1.position - p2.position).length() < 1.5 * (p1.radius + p2.radius)) {
                    merge_particle(i, j);
                    flag = 1;
                }
            }
        }
        if (flag) {
            check_collapse();
        }
    }

    // 计算所有质点间的引力
    void computeForces() {
        // const int n = particles.size();
        
        // 重置加速度
        for (int k = 0; k < index; k++){
            auto& p = particles[k];
            p.prev_acceleration = p.acceleration;
            p.acceleration = Vec3();
        }

        // for (auto& p : particles) {
        //     p.prev_acceleration = p.acceleration;
        //     p.acceleration = Vec3();
        // }
        
        // 计算每对质点间的引力
        check_collapse();
        for (int i = 0; i < index; ++i) {
            for (int j = i+1; j < index; ++j) {
                Vec3 delta = particles[j].position - particles[i].position;
                double dist = delta.length();
                double dist_cubed = (dist*dist + softening*softening) * std::sqrt(dist*dist + softening*softening);
                double force_magnitude = G * particles[i].mass * particles[j].mass / dist_cubed;
                
                Vec3 force = delta * force_magnitude;
                
                particles[i].acceleration = particles[i].acceleration + force * (1.0 / particles[i].mass);
                particles[j].acceleration = particles[j].acceleration - force * (1.0 / particles[j].mass);
            }
        }
        for (int i = 0; i < index; i++){
            auto& p = particles[i];
            p.acceleration = p.acceleration - p.position * G * M * (1.0/ pow(p.position.length(), 3));
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
    size_t size() const { return index; }
    
    // 获取质点位置
    const Vec3& getPosition(int i) const { return particles[i].position; }

    const Particle& getParticle(int i) const { return particles[i]; }

    void print(){
        for (int i = 0; i < index; i++){
            auto p = particles[i];
            std::cout << " x " <<p.position.x <<" y "<< p.position.y <<" z "<< p.position.z <<" m "<< p.mass <<" r "<< p.radius << " v " << p.velocity.length() << std::endl;
        }
    }
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
    const double G = 1;  // 万有引力常数
    const double softening = 1.0;  // 软化长度
    const double dt = 0.1;  // 时间步长
    const int steps = 100000;  // 模拟步数
    // const double box_size = 100.0;  // 初始分布区域大小
    const double M = 1000;
    const double R = 10000;
    const double r = 100;
    
    NBodySystem system(G, M, softening);
    
    // 初始化随机质点
    for (int i = 0; i < N; ++i) {
        // Vec3 pos(randomDouble(-box_size, box_size), 
        //         randomDouble(-box_size, box_size), 
        //         randomDouble(-box_size, box_size));
        // Vec3 vel(randomDouble(-1, 1), 
        //         randomDouble(-1, 1), 
        //         randomDouble(-1, 1));
        // Vec3 vel(0, 
        //         0, 
        //         0);
        double mass = randomDouble(0.1, 0.2);
        double radius = randomDouble(1.0, 5.0);
        
        system.addParticle(sample(R, r, G, M, mass, radius));
    }
    
    // 打开输出文件
    std::ofstream outfile("trajectories.txt");
    
    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 模拟循环
    for (int step = 0; step < steps; ++step) {
        system.simulateStep(dt);
        
        // 输出当前位置到文件
        if (step % 1 == 0){
            for (int i = 0; i < system.size(); ++i) {
                const Vec3& pos = system.getPosition(i);
                const Particle& p = system.getParticle(i);
                outfile << pos.x << "," << pos.y << "," << pos.z << "," << p.radius << ";";
            }
        }

        outfile << "\n";
        
        // 打印进度
        if (step % 1000 == 0) {
            std::cout << system.size();
            std::cout << "Step " << step << "/" << steps << " completed\n";
            system.print();
        }
    }
    
    // 结束计时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Simulation completed in " << duration.count() << " ms\n";
    system.print();

    outfile.close();
    
    std::cout << "Trajectories saved to trajectories.txt\n";
    
    return 0;
}