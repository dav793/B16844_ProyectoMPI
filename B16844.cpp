/**
 *  Usage:      ./sim 
 *                  <number of people> 
 *                  <rate of infection> 
 *                  <virus duration> 
 *                  <chance of recovery> 
 *                  <chance of death> 
 *                  <starting infected> 
 *                  <world size> 
 *                  <initial ticks>
 * 
 *  Prep:       export TMPDIR=/tmp
 *  Compile:    $HOME/opt/usr/local/bin/mpic++ -std=c++11 -o B16844 ./B16844.cpp
 *  Exec:       $HOME/opt/usr/local/bin/mpiexec -np 4 ./B16844 1000 0.65 20 0.50 0 20 100 1000
 * 
 *  Restricciones:
 *      - numero de personas debe ser multiplo de cantidad de procesos (de lo contrario se pueden perder algunas personas)
 */

#include <mpi.h>
#include <vector>
#include <tuple>
#include <random>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip> // std::setw, setprecision
#include "tick-log.h"
using namespace std;

int getArgs(int argc, char* argv[], int& numberPeople, double& infectiousness, int& virusDuration, double& chanceRecover, double& chanceDeath, int& startingInfected, int& worldSize, int& initialTicks);
int* createPeople(int numberPeople, int startingInfected, int worldSize);
int* createWorld(int worldSize, int numberPeople, int* people);
void syncWorld(int worldSize, int* local_world, int* global_world);
int getWorldIndexFromPosition(int x, int y, int worldSize);
void syncWorldCell(int cellX, int cellY, int worldSize, int* local_world, int* global_world);
float getRandomFloat();
int getRandomIntInRange(int min, int max);
void generateOutputFile(string filename, vector<TickLog*>& tickLog);

std::random_device rd; // obtain a random number from hardware
std::mt19937 eng(rd()); // seed the generator
std::uniform_int_distribution<> distr(0, 1000); // define the range

const int STATE_HEALTHY = 0;
const int STATE_INFECTED = 1;
const int STATE_IMMUNE = 2;
const int STATE_DEAD = 3;

const bool IS_SILENT = true;
// const bool IS_SILENT = false;

const bool NOTIFY_ON_TICK_END = true;
// const bool NOTIFY_ON_TICK_END = false;

const string OUTPUT_FILENAME("out");

const bool PRODUCE_OUTPUT_FILE = true;
// const bool PRODUCE_OUTPUT_FILE = false;

int main(int argc, char* argv[]) {

    int rank;
    int proc_qty;
    MPI_Status mpi_status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_qty);

    int numberPeople;
    double infectiousness;
    int virusDuration;
    double chanceRecover;
    double chanceDeath;
    int startingInfected;
    int worldSize;
    int initialTicks;
    bool finished = false;

    int currentTick = 0;
    int currentHealthy = 0;
    int currentInfected = 0;
    int currentImmune = 0;
    int currentDead = 0;

    vector<TickLog*> tickLog;

    // a person descriptor is represented by the following set of consecutive integers:
    //  0: pos x
    //  1: pos y
    //  2: state
    //  3: last state since

    // a cell descriptor is represented by the following set of consecutive integers:
    //  0: number of healthy people at this cell
    //  1: number of infected people at this cell
    //  2: number of immune people at this cell
    //  3: number of dead people at this cell
    // a cell descriptor's index in the world array determines the cell's position in the world.

    getArgs(argc, argv, numberPeople, infectiousness, virusDuration, chanceRecover, chanceDeath, startingInfected, worldSize, initialTicks);

    MPI_Barrier(MPI_COMM_WORLD);
    double local_start = MPI_Wtime();

    int people_per_proc = numberPeople / proc_qty;

    int* global_people = NULL;
    int* global_world = new int[worldSize*worldSize*4];

    // root process creates all people and entire world
    if (rank == 0) {
        global_people = createPeople(numberPeople, startingInfected, worldSize);
        global_world = createWorld(worldSize, numberPeople, global_people);

        currentInfected = startingInfected;
        currentHealthy = numberPeople - startingInfected;
    }

    // distribute people among processes
    int* local_people = new int[people_per_proc*4];
    MPI_Scatter(global_people, people_per_proc*4, MPI_INT, local_people, people_per_proc*4, MPI_INT, 0, MPI_COMM_WORLD);

    // distribute world among processes
    MPI_Bcast(global_world, worldSize*worldSize*4, MPI_INT, 0, MPI_COMM_WORLD);

    // execute ticks
    for (int i = 0; i < initialTicks; ++i) {

        // each element in vector is a tuple of 3 elements: < person index, previous state, new state >
        vector< tuple< int, int, int > > state_changes;

        // process infection
        for (int j = 0; j < people_per_proc; ++j) {

            int person_x = local_people[ j*4 + 0 ];
            int person_y = local_people[ j*4 + 1 ];
            int person_state = local_people[ j*4 + 2 ];
            int person_last_state_since = local_people[ j*4 + 3 ];

            if (person_state == STATE_INFECTED) {

                if (currentTick - person_last_state_since >= virusDuration) { // person has been infected as long as it can

                    int result = getRandomIntInRange(0, 101);
                    double chanceToRecover = chanceRecover * 100;
                    if (result <= chanceToRecover) { // person recovers

                        local_people[ j*4 + 2 ] = STATE_IMMUNE;
                        local_people[ j*4 + 3 ] = currentTick;

                        int global_person_index = (rank * people_per_proc * 4) + (j * 4);
                        tuple< int, int, int > change = make_tuple(global_person_index, STATE_INFECTED, STATE_IMMUNE);
                        state_changes.push_back(change);

                        if (!IS_SILENT)
                            printf("NOTICE: person at (%d, %d) recovered during tick %d\n", person_x, person_y, currentTick);

                    }
                    else {  // person dies

                        local_people[ j*4 + 2 ] = STATE_DEAD;
                        local_people[ j*4 + 3 ] = currentTick;

                        int global_person_index = (rank * people_per_proc * 4) + (j * 4);
                        tuple< int, int, int > change = make_tuple(global_person_index, STATE_INFECTED, STATE_DEAD);
                        state_changes.push_back(change);

                        if (!IS_SILENT)
                            printf("NOTICE: person at (%d, %d) died during tick %d\n", person_x, person_y, currentTick);

                    }

                }

            }

        }


        // spread infection
        for (int j = 0; j < people_per_proc; ++j) {

            int person_x = local_people[ j*4 + 0 ];
            int person_y = local_people[ j*4 + 1 ];
            int person_state = local_people[ j*4 + 2 ];

            if (person_state == STATE_HEALTHY) {

                int world_index = getWorldIndexFromPosition(person_x, person_y, worldSize);
                int infectedAtCell = global_world[ world_index + STATE_INFECTED ];
                if (infectedAtCell > 0) {

                    bool infect = false;
                    for (int k = 0; k < infectedAtCell; ++k) {
                        int result = getRandomIntInRange(0, 101);
                        double chanceToBeInfected = infectiousness * 100;
                        if (result <= chanceToBeInfected)
                            infect = true;
                    }

                    if (infect) {   // infect person

                        local_people[ j*4 + 2 ] = STATE_INFECTED;
                        local_people[ j*4 + 3 ] = currentTick;

                        int global_person_index = (rank * people_per_proc * 4) + (j * 4);
                        tuple< int, int, int > change = make_tuple(global_person_index, STATE_HEALTHY, STATE_INFECTED);
                        state_changes.push_back(change);

                        if (!IS_SILENT)
                            printf("NOTICE: person at (%d, %d) became infected during tick %d\n", person_x, person_y, currentTick);

                    }

                }

            }

        }


        // each process sends local changes size to root process
        int local_changes_size = state_changes.size();
        MPI_Send(&local_changes_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);    

        // root proc compiles all local change sizes into a single array of change sizes
        int* global_changes_sizes = NULL;
        if (rank == 0) {
            global_changes_sizes =  new int[proc_qty];

            for (int j = 0; j < proc_qty; ++j) {
                MPI_Recv(&global_changes_sizes[j], 1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
            }
        }

        // each proccess converts local changes vector to array for sending
        int* local_changes = new int[local_changes_size * 3];
        for (int j = 0; j < local_changes_size; ++j) {
            local_changes[j*3 + 0] = get<0>(state_changes[j]);
            local_changes[j*3 + 1] = get<1>(state_changes[j]);
            local_changes[j*3 + 2] = get<2>(state_changes[j]);
        }

        // each process sends local changes to root process  
        if (rank != 0) {
            MPI_Send(local_changes, local_changes_size * 3, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }   

        if (rank == 0) {
            int** global_changes = new int*[proc_qty];

            global_changes[0] = new int[global_changes_sizes[0] * 3];
            MPI_Sendrecv(local_changes, local_changes_size * 3, MPI_INT,
                0, 0,
                global_changes[0], global_changes_sizes[0] * 3, MPI_INT,
                0, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int j = 1; j < proc_qty; ++j) {
                global_changes[j] = new int[global_changes_sizes[j] * 3];

                // root process compiles all local changes into one global changes 2-dimensional array
                MPI_Recv(global_changes[j], global_changes_sizes[j] * 3, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // root process commits changes to own global world
            for (int j = 0; j < proc_qty; ++j) {
                for (int k = 0; k < global_changes_sizes[j]; ++k) {

                    int person_index = global_changes[j][k*3+0];
                    int previous_status = global_changes[j][k*3+1];
                    int next_status = global_changes[j][k*3+2];

                    global_people[person_index+2] = next_status;
                    global_people[person_index+3] = currentTick;
                    
                    int person_x = global_people[person_index+0];
                    int person_y = global_people[person_index+1];
                    int world_index = getWorldIndexFromPosition(person_x, person_y, worldSize);

                    global_world[ world_index + previous_status ] -= 1;
                    global_world[ world_index + next_status ] += 1;

                    switch (previous_status) {
                        case STATE_HEALTHY:
                            currentHealthy--;
                            break;
                        case STATE_INFECTED:
                            currentInfected--;
                            break;
                        case STATE_IMMUNE:
                            currentImmune--;
                            break;
                        case STATE_DEAD:
                            currentDead--;
                            break;
                    }

                    switch (next_status) {
                        case STATE_HEALTHY:
                            currentHealthy++;
                            break;
                        case STATE_INFECTED:
                            currentInfected++;
                            break;
                        case STATE_IMMUNE:
                            currentImmune++;
                            break;
                        case STATE_DEAD:
                            currentDead++;
                            break;
                    }

                }
            }

            // root process disposes of changes array
            for (int j = 0; j < proc_qty; ++j) {
                delete [] global_changes[j];
            }
            delete [] global_changes;
        }

        delete [] local_changes;
        if (rank == 0) {
            delete [] global_changes_sizes;
        }


        // move people
        int* local_movements = new int[people_per_proc * 3];
        for (int j = 0; j < people_per_proc; ++j) {

            int person_x = local_people[ j*4 + 0 ];
            int person_y = local_people[ j*4 + 1 ];
            int person_state = local_people[ j*4 + 2 ];
            
            int global_person_index = (rank * people_per_proc * 4) + (j * 4);
            int new_x = person_x;
            int new_y = person_y;
            if (person_state != STATE_DEAD) {
                int movement_x = getRandomIntInRange(0, 3) - 1;
                int movement_y = getRandomIntInRange(0, 3) - 1;
                new_x = person_x + movement_x;
                new_y = person_y + movement_y;
            }

            if (new_x < 0)
                new_x = worldSize-1;
            if (new_x >= worldSize)
                new_x = 0;

            if (new_y < 0)
                new_y = worldSize-1;
            if (new_y >= worldSize)
                new_y = 0;

            local_people[ j*4 + 0 ] = new_x;
            local_people[ j*4 + 1 ] = new_y;

            local_movements[j*3 + 0] = global_person_index;
            local_movements[j*3 + 1] = new_x;
            local_movements[j*3 + 2] = new_y;

        }

        
        int* global_movements = NULL;
        if (rank == 0)
            global_movements = new int[people_per_proc * proc_qty * 3];       

        // each process sends local movement changes to root process 
        MPI_Gather(local_movements, people_per_proc * 3, MPI_INT, global_movements, people_per_proc * 3, MPI_INT, 0, MPI_COMM_WORLD);
        
        // if (rank == 0) {
        //     for (int j = 0; j < people_per_proc * proc_qty; ++j) {
        //         printf("person %d moved to (%d, %d)\n", global_movements[j*3+0], global_movements[j*3+1], global_movements[j*3+2]);
        //     }
        // }

        if (rank == 0) {
            for (int j = 0; j < people_per_proc * proc_qty; ++j) {

                int person_index = global_movements[j*3+0];

                int person_x = global_people[person_index+0];
                int person_y = global_people[person_index+1];
                int person_status = global_people[person_index+2];
                int previous_world_index = getWorldIndexFromPosition(person_x, person_y, worldSize);

                int new_person_x = global_movements[j*3+1];
                int new_person_y = global_movements[j*3+2];
                int new_world_index = getWorldIndexFromPosition(new_person_x, new_person_y, worldSize);

                global_people[person_index+0] = new_person_x;
                global_people[person_index+1] = new_person_y;

                global_world[ previous_world_index + person_status ] -= 1;
                global_world[ new_world_index + person_status ] += 1;

            }
        }

        delete [] local_movements;
        if (rank == 0)
            delete [] global_movements;

        // update person counters
        MPI_Bcast(&currentHealthy, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&currentInfected, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&currentImmune, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&currentDead, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // log tick stats
        if (rank == 0) {
            TickLog* current = new TickLog(currentTick, currentHealthy, currentInfected, currentImmune, currentDead);
            tickLog.push_back(current);
        }

        // broadcast root process' world to every process
        MPI_Bcast(global_world, worldSize * worldSize * 4, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0 && NOTIFY_ON_TICK_END)
            printf("tick %d done.\n", currentTick);

        // if there are no more infected, end simulation
        if (currentInfected == 0) {
            if (rank == 0 && !IS_SILENT)
                printf("NOTICE: Infection was erradicated during tick %d\n", currentTick);
            break;
        }
        
        currentTick++;

    }

    double local_finish = MPI_Wtime();
    double local_elapsed = local_finish - local_start;
    double global_elapsed;
    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // print results
    if (rank == 0) {
        printf("Execution time: %f\n", global_elapsed);
    }

    // generate output file
    if (rank == 0 && PRODUCE_OUTPUT_FILE) {
        generateOutputFile(OUTPUT_FILENAME, tickLog);
        cout << "Generated output file: " << OUTPUT_FILENAME << ".txt" << endl;
    }

    // clean up
    if (rank == 0) {
        delete [] global_people;
    }
    delete [] global_world;
    delete [] local_people;

    MPI_Barrier(MPI_COMM_WORLD); // para sincronizar la finalizaci�n de los procesos
    MPI_Finalize();
    return 0;
}

int getArgs(int argc, char* argv[], int& numberPeople, double& infectiousness, int& virusDuration, double& chanceRecover, double& chanceDeath, int& startingInfected, int& worldSize, int& initialTicks) {

    if (argc < 9) {
        cout << "Usage: ./sim <number of people> <rate of infection> <virus duration> <chance of recovery> <chance of death> <starting infected> <world size> <initial ticks>" << endl;
        return -1;
    }
        
    numberPeople = atoi(argv[1]);
    infectiousness = atof(argv[2]);
    virusDuration = atof(argv[3]);
    chanceRecover = atof(argv[4]);
    chanceDeath = atof(argv[5]);
    startingInfected = atoi(argv[6]);
    worldSize = atoi(argv[7]);
    initialTicks = atoi(argv[8]);

    return 0;

}

void tick() {

}

int* createPeople(int numberPeople, int startingInfected, int worldSize) {

    int* people = new int[numberPeople * 4];
    int currentNumberInfected = 0;

    for (int i = 0; i < numberPeople; ++i) {

        int x = getRandomIntInRange(0, worldSize);
        int y = getRandomIntInRange(0, worldSize);

        int state = STATE_HEALTHY;
        if (currentNumberInfected < startingInfected) {
            state = STATE_INFECTED;
            currentNumberInfected++;
        }

        people[ i*4 + 0 ] = x;      // pos x
        people[ i*4 + 1 ] = y;      // pos y
        people[ i*4 + 2 ] = state;  // state
        people[ i*4 + 3 ] = 0;      // last state since

    }

    return people;

}

int* createWorld(int worldSize, int numberPeople, int* people) {

    int* world = new int[worldSize * worldSize * 4];
    for (int i = 0; i < worldSize * worldSize * 4; ++i) {
        world[i] = 0;
    }

    for (int i = 0; i < numberPeople; ++i) {

        int person_x = people[i*4 + 0];
        int person_y = people[i*4 + 1];
        int person_state = people[i*4 + 2];

        int world_index = getWorldIndexFromPosition(person_x, person_y, worldSize);
        world[world_index + person_state] += 1;

    }

    return world;

}

// reduce all local world stats into a combined global world, of which all processes have an exact copy.
void syncWorld(int worldSize, int* local_world, int* global_world) {
    for (int i = 0; i < worldSize * worldSize; ++i) {
        MPI_Allreduce(&local_world[i*4+0], &global_world[i*4+0], 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_world[i*4+1], &global_world[i*4+1], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_world[i*4+2], &global_world[i*4+2], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_world[i*4+3], &global_world[i*4+3], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
}

// reduce a single local cell stats into a combined global cell, of which all processes have an exact copy.
void syncWorldCell(int cellX, int cellY, int worldSize, int* local_world, int* global_world) {
    int world_index = getWorldIndexFromPosition(cellX, cellY, worldSize);

    MPI_Allreduce(&local_world[world_index+0], &global_world[world_index+0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_world[world_index+1], &global_world[world_index+1], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_world[world_index+2], &global_world[world_index+2], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_world[world_index+3], &global_world[world_index+3], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

int getWorldIndexFromPosition(int x, int y, int worldSize) {
    return (x * 4) + (y * worldSize * 4);
}

// generate random float between 0 and 1
float getRandomFloat() {
    return ((float) distr(eng)) / 1000;
}

// generate random int between <min> (inclusive) and <max> (exclusive)
int getRandomIntInRange(int min, int max) {
    return floor(getRandomFloat() * (max-min-1) + min);
}

void generateOutputFile(string filename, vector<TickLog*>& tickLog) {

    std::ofstream outfile (filename + ".txt");

    int totalHealthy = (* tickLog[ tickLog.size()-1 ]).getHealthyCount();
    int totalInfected = (* tickLog[ tickLog.size()-1 ]).getInfectedCount();
    int totalImmune = (* tickLog[ tickLog.size()-1 ]).getImmuneCount();
    int totalDead = (* tickLog[ tickLog.size()-1 ]).getDeadCount();
    int totalPeople = totalHealthy + totalInfected + totalImmune + totalDead;

    double healthyPerTick;
    double healthyPercentage = (double) totalHealthy / (double) totalPeople;

    double infectedPerTick;
    double infectedPercentage = (double) totalInfected / (double) totalPeople;

    double immunePerTick;
    double immunePercentage = (double) totalImmune / (double) totalPeople;

    double deadPerTick;
    double deadPercentage = (double) totalDead / (double) totalPeople;

    outfile << endl;
    outfile << "FINAL STATISTICS: " << endl << endl;

    outfile << setw(20) << left << "- Current tick: " << (* tickLog[ tickLog.size()-1 ]).getTickNumber() << endl;
    outfile << setw(20) << left << "- Total people: " << totalPeople << endl;
    outfile << setw(20) << left << "- Healthy: " << totalHealthy << " (" << healthyPercentage*100 << "%)" << endl;
    outfile << setw(20) << left << "- Infected: " << totalInfected << " (" << infectedPercentage*100 << "%)" << endl;
    outfile << setw(20) << left << "- Immune: " << totalImmune << " (" << immunePercentage*100 << "%)" << endl;
    outfile << setw(20) << left << "- Dead: " << totalDead << " (" << deadPercentage*100 << "%)" << endl;

    outfile << endl << endl;

    outfile << "PER-TICK STATISTICS: " << endl << endl;

    for (int i = 0; i < tickLog.size(); ++i) {
        TickLog* entry = tickLog[i];

        int tickTotalHealthy = (* entry).getHealthyCount();
        int tickTotalInfected = (* entry).getInfectedCount();
        int tickTotalImmune = (* entry).getImmuneCount();
        int tickTotalDead = (* entry).getDeadCount();

        double healthyPercentage = (double) tickTotalHealthy / (double) totalPeople * 100;
        double infectedPercentage = (double) tickTotalInfected / (double) totalPeople * 100;
        double immunePercentage = (double) tickTotalImmune / (double) totalPeople * 100;
        double deadPercentage = (double) tickTotalDead / (double) totalPeople * 100;

        stringstream streamA;
        streamA << fixed << setprecision(2) << healthyPercentage;
        string healthyPercentageStr = streamA.str();

        stringstream streamB;
        streamB << fixed << setprecision(2) << infectedPercentage;
        string infectedPercentageStr = streamB.str();

        stringstream streamC;
        streamC << fixed << setprecision(2) << immunePercentage;
        string immunePercentageStr = streamC.str();

        stringstream streamD;
        streamD << fixed << setprecision(2) << deadPercentage;
        string deadPercentageStr = streamD.str();

        double healthyAvg = tickTotalHealthy;
        double infectedAvg = tickTotalInfected;
        double immuneAvg = tickTotalImmune;
        double deadAvg = tickTotalDead;

        for (int j = 0; j < i; ++j) {
            healthyAvg += (* tickLog[j]).getHealthyCount();
            infectedAvg += (* tickLog[j]).getInfectedCount();
            immuneAvg += (* tickLog[j]).getImmuneCount();
            deadAvg += (* tickLog[j]).getDeadCount();
        }

        if (i > 0) {
            healthyAvg = healthyAvg / (double) (i+1);
            infectedAvg = infectedAvg / (double) (i+1);
            immuneAvg = immuneAvg / (double) (i+1);
            deadAvg = deadAvg / (double) (i+1);
        }

        outfile << setw(20) << left << "TICK " + to_string((* entry).getTickNumber()) + ": " << setw(15) << left << "Total" << setw(15) << left << "Percentage" << setw(15) << left << "Average" << endl;
        outfile << setw(20) << left << "- Healthy: " << setw(15) << left << tickTotalHealthy << setw(15) << left << healthyPercentageStr + "%" << setw(15) << left << healthyAvg << endl;
        outfile << setw(20) << left << "- Infected: " << setw(15) << left << tickTotalInfected << setw(15) << left << infectedPercentageStr + "%" << setw(15) << left << infectedAvg << endl;
        outfile << setw(20) << left << "- Immune: " << setw(15) << left << tickTotalImmune << setw(15) << left << immunePercentageStr + "%" << setw(15) << left << immuneAvg << endl;
        outfile << setw(20) << left << "- Dead: " << setw(15) << left << tickTotalDead << setw(15) << left << deadPercentageStr + "%" << setw(15) << left << deadAvg << endl;

        outfile << endl;
    }

    outfile.close();

}