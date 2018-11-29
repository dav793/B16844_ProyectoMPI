#ifndef TICK_LOG_H
#define TICK_LOG_H

class TickLog {

    public:
        TickLog(int tick, int healthyCount, int infectedCount, int immuneCount, int deadCount);
        int getTickNumber();
        int getHealthyCount();
        int getInfectedCount();
        int getImmuneCount();
        int getDeadCount();

    private:
        int _tick;
        int _healthyCount;
        int _infectedCount;
        int _immuneCount;
        int _deadCount;

};

TickLog::TickLog(int tick, int healthyCount, int infectedCount, int immuneCount, int deadCount) {
    _tick = tick;
    _healthyCount = healthyCount;
    _infectedCount = infectedCount;
    _immuneCount = immuneCount;
    _deadCount = deadCount;
}

int TickLog::getTickNumber() {
    return _tick;
}

int TickLog::getHealthyCount() {
    return _healthyCount;
}

int TickLog::getInfectedCount() {
    return _infectedCount;
}

int TickLog::getImmuneCount() {
    return _immuneCount;
}

int TickLog::getDeadCount() {
    return _deadCount;
}

#endif /* TICK_LOG_H */