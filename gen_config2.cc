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
    int count = 180;
    
    string three_layers[] = {"2 3 2", "4 5 4", "10 15 10"};
    string four_layers[] = {"2 3 3 2", "4 5 5 4", "10 15 15 10"};
    string five_layers[] = {"2 3 4 3 2", "4 5 6 5 4", "10 15 20 15 10"};
    
    float learning_rates[] = {0.1, 0.25, 0.5, 1.0, 2.0};
    
    int epochs[] = {1000, 10000, 40000, 100000};
    
    int num_inputs = 3;
    
    //Number of layers
    for(int a = 3; a < 6; a++)
    {
        //Number of Neurons
        for(int b = 0; b < 3; b++)
        {
            //Learning Rate
            for(int c = 0; c < 5; c++)
            {
                //Number of epcohs
                for(int d = 0; d < 4; d++)
                {
                    //Open and name config file
                    char filename[100];
                    sprintf(filename, "config/config%d.cf", count++);
                    ofstream outfile(filename);
                    
                    //Make sure config file opened
                    if (!outfile.is_open())
                    {
                        cerr << "Cannot open " << filename << endl;
                        exit(1);
                    }
                    
                    //Print number of layers
                    outfile << a << endl;
                    
                    //Print Number of Neurons
                    if (a == 3)
                        outfile << three_layers[b] << endl;
                    if (a == 4)
                        outfile << four_layers[b] << endl;
                    if (a == 5)
                        outfile << five_layers[b] << endl;
                    
                    //Print Learning Rate
                    outfile << learning_rates[c] << endl;
                    
                    outfile << "lab4/training2.txt" << endl;
                    outfile << "lab4/testing2.txt" << endl;
                    outfile << "lab4/validation2.txt" << endl;
                    
                    //Print number of epcohs
                    outfile << epochs[d] << endl;
                    
                    //Print number of inputs
                    outfile << num_inputs << endl;    
                }
                
            }
        }
    }
    
    return 1;
}