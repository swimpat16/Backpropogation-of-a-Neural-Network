#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <ctime>
using namespace std;

int main()
{
    for (int i = 0; i < 360; i++)
    {
        char command[1000];
        sprintf(command, "lab4/lab4 out/out%d.csv < config/config%d.cf > /dev/null", i, i);
        
        system(command);
        
        printf("Finished config %3d / 360\n", i + 1);
    }
}