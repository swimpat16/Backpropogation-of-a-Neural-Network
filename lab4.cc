/*      lab4.cc
 *      Alex Chaloux and Patrick Elam
 *      3/30/15
 */

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


int HIDDEN_LAYERS;
vector<int> NUM_NEURONS;    // Number of neurons for hidden layer i
double LEARNING_RATE;
string TRAINING_FILE;
string TESTING_FILE;
string VALIDATION_FILE;
int NUM_EPOCHS;
int NUM_INPUTS;

vector< vector< vector<double> > > hiddenweights;   // Weight values of linkages
vector< vector<double> > hiddenbiases;  // Bias values of neurons in hidden layers
vector<double> outputweights;           // Weights of linkages to output node
vector< vector<double> > hiddenvalues;  // Sigma values for neurons
double outputbias;                      // Bias value of output node
vector< vector<double> > deltahidden;   // Delta values of neurons in hidden layer
double deltaoutput;                     // Delta value of output node
vector<double> testresults;             // Results for each epoch
double validationrmse;                  // Final RMSE from validation file


void init_weights()
{
    //Holds results for each epoch
    testresults.resize(NUM_EPOCHS);
    
    //3D, weights of linkages between nodes
    hiddenweights.resize(HIDDEN_LAYERS);
    
    //2D, properties of neurons
    hiddenvalues.resize(HIDDEN_LAYERS);
    hiddenbiases.resize(HIDDEN_LAYERS);
    deltahidden.resize(HIDDEN_LAYERS);
    
    for (int i = 0; i < HIDDEN_LAYERS; i++)
    {
        //Resize for number of neurons in given layer
        hiddenweights[i].resize(NUM_NEURONS[i]);
        hiddenvalues[i].resize(NUM_NEURONS[i]);
        hiddenbiases[i].resize(NUM_NEURONS[i]);
        deltahidden[i].resize(NUM_NEURONS[i]);
        
        for (int j = 0; j < NUM_NEURONS[i]; j++)
        {
            //If first hidden layer
            if(i == 0)
            {
                //Resize for number of inputs
                hiddenweights[i][j].resize(NUM_INPUTS);
            }
            else
            {   
                //Resize for number of neurons on last layer
                hiddenweights[i][j].resize(NUM_NEURONS[i-1]);
            }
            
            //Assign biases to neurons
            hiddenbiases[i][j] = (rand() * 1.0) / (RAND_MAX * 1.0);
            hiddenbiases[i][j] /= 5.0;
            hiddenbiases[i][j] -= 0.1;
        }
    }
    
    // Initialize weights to random number between -0.1 to 0.1
    for (int i = 0; i < hiddenweights.size(); i++)
    {
        for (int j = 0; j < hiddenweights[i].size(); j++)
        {
            for (int k = 0; k < hiddenweights[i][j].size(); k++)
            {
                hiddenweights[i][j][k] = (rand() * 1.0) / (RAND_MAX * 1.0);
                hiddenweights[i][j][k] /= 5.0;
                hiddenweights[i][j][k] -= 0.1;
            }
        }
    }
    
    // Resize outputweight vector, which is the weights of the linkages to the outputs from the last layer
    outputweights.resize(hiddenweights[HIDDEN_LAYERS - 1].size());
    
    // Randomly initialize 
    for (int i = 0; i < outputweights.size(); i++)
    {
        outputweights[i] = (rand() * 1.0) / (RAND_MAX * 1.0);
        outputweights[i] /= 5.0;
        outputweights[i] -= 0.1;
    }
    
    // Randomly initialize output biases
    outputbias = (rand() * 1.0) / (RAND_MAX * 1.0);
    outputbias /= 5.0;
    outputbias -= 0.1;
}

void train_net()
{
    
    // read in training file
    ifstream infile(TRAINING_FILE.c_str());
    if (!infile.is_open())
    {
        cerr << "Invalid training file" << endl;
        exit(1);
    }
    
    vector<float> training_inputs;
    training_inputs.resize(NUM_INPUTS);
    float training_output;
    float outputvalue;
    
    // Each iteration of the loop will read one full line of the input file
    while (infile >> training_inputs[0])
    {
        // Reads the rest of the line, as the previous line just got the first input
        for (int i = 1; i < NUM_INPUTS; i++)
        {
            // Reads the input lines
            infile >> training_inputs[i];
        }
        //Reads the expected output
        infile >> training_output;
        
        
        // "Computing the Outputs" section
        for(int i = 0; i < hiddenweights.size(); i++)
        {
            // For each neuron:
            for(int j = 0; j < hiddenweights[i].size(); j++)
            {
                // Initialize sigma value to 0
                hiddenvalues[i][j] = 0.0;
                
                // If first hidden layer
                if(i == 0)
                {
                    // Update for number of inputs
                    for (int k = 0; k < NUM_INPUTS; k++)
                    {
                        hiddenvalues[i][j] += hiddenweights[i][j][k] * training_inputs[k];
                    }
                }
                else
                {
                    // Update for number of previous hidden layer
                    for (int k = 0; k < hiddenweights[i][j].size(); k++)
                    {
// ?
                        hiddenvalues[i][j] += hiddenweights[i][j][k] * hiddenvalues[i-1][k];
                    }
                }
                
                // Add neuron's bias to output
                hiddenvalues[i][j] += hiddenbiases[i][j];
                
                // Calculate the sigma value
                hiddenvalues[i][j] = 1.0 / (1.0 + exp(0.0 - hiddenvalues[i][j]));
            }
        }
        
        // Reinitialize output value
        outputvalue = 0.0;
        
        // For each linkage to output node
        for (int i = 0; i < hiddenvalues[hiddenvalues.size() - 1].size(); i++)
        {
            outputvalue += outputweights[i] * hiddenvalues[hiddenvalues.size() - 1][i];
        }
        
        // Add output's bias
        outputvalue += outputbias;
        
        // Calculate the sigma value
        outputvalue = 1.0 / (1.0 + exp(0.0 - outputvalue));
        
        
        
        // "Computing the Delta Values" section
        // Compute delta value of output node
        deltaoutput = outputvalue * (1.0 - outputvalue) * (training_output - outputvalue);
        
        
        for(int i = hiddenweights.size() - 1; i >= 0; i--)
        {
            for(int j = 0; j < hiddenweights[i].size(); j++)
            {
                // If last layer in hidden layers
                if(i == hiddenweights.size() - 1)
                {
                    deltahidden[i][j] = hiddenvalues[i][j] * (1 - hiddenvalues[i][j]) * deltaoutput * outputweights[j];
                }
                else
                {
                    deltahidden[i][j] = 0.0;
                    for (int k = 0; k < hiddenweights[i + 1].size(); k++)
                    {
                        deltahidden[i][j] += hiddenvalues[i][j] * (1.0 - hiddenvalues[i][j]) * deltahidden[i+1][k] * hiddenweights[i+1][k][j];
                    }
                }
                    
            }
        }
        
        
    
        // "Updating the Weights" section
        for(int i = 0; i < hiddenweights.size(); i++)
        {
            for(int j = 0; j < hiddenweights[i].size(); j++)
            {
                if (i == 0)
                {
                    for (int k = 0; k < NUM_INPUTS; k++)
                    {
                        hiddenweights[i][j][k] += LEARNING_RATE * deltahidden[i][j] * training_inputs[k];
                    }
                }
                else
                {
                    for (int k = 0; k < hiddenweights[i][j].size(); k++)
                    {
                        hiddenweights[i][j][k] += LEARNING_RATE * deltahidden[i][j] * hiddenvalues[i - 1][k];
                    }
                }
                
                // Update the neuron bias
                hiddenbiases[i][j] += LEARNING_RATE * deltahidden[i][j];
            }
        }
        
        // Update the output weights
        for (int i = 0; i < outputweights.size(); i++)
        {
            outputweights[i] += LEARNING_RATE * deltaoutput * hiddenvalues[hiddenvalues.size() - 1][i];
        }
        
        // Update the output bias
        outputbias += LEARNING_RATE * deltaoutput;
    }
    
    infile.close();
    
    
    
}

void test_net(int index)
{
    // read in testing file
    ifstream infile(TESTING_FILE.c_str());
    if (!infile.is_open())
    {
        cerr << "Invalid testing file" << endl;
        exit(1);
    }
    
    vector<float> testing_inputs;
    testing_inputs.resize(NUM_INPUTS);
    float testing_output;
    float outputvalue;
    float testingsum = 0;
    int numlines = 0;
    float testrmse;
    
    while (infile >> testing_inputs[0])
    {
        numlines++;
        
        for (int i = 1; i < NUM_INPUTS; i++)
        {
            infile >> testing_inputs[i];
        }
        infile >> testing_output;
        
        // compute outputs
        for(int i = 0; i < hiddenweights.size(); i++)
        {
            for(int j = 0; j < hiddenweights[i].size(); j++)
            {
                hiddenvalues[i][j] = 0.0;
                if(i == 0)
                {
                    // go through inputs
                    for (int k = 0; k < NUM_INPUTS; k++)
                    {
                        hiddenvalues[i][j] += hiddenweights[i][j][k] * testing_inputs[k];
                    }
                }
                else
                {
                    // go through previous hidden layer
                    for (int k = 0; k < hiddenweights[i][j].size(); k++)
                    {
                        hiddenvalues[i][j] += hiddenweights[i][j][k] * hiddenvalues[i-1][k];
                    }
                }
                
                hiddenvalues[i][j] += hiddenbiases[i][j];
                hiddenvalues[i][j] = 1.0 / (1.0 + exp(0.0 - hiddenvalues[i][j]));
            }
        }
        
        outputvalue = 0.0;
        for (int i = 0; i < hiddenvalues[hiddenvalues.size() - 1].size(); i++)
        {
            outputvalue += outputweights[i] * hiddenvalues[hiddenvalues.size() - 1][i];
        }
        outputvalue += outputbias;
        outputvalue = 1.0 /(1.0 + exp(0.0 - outputvalue));
        
        testingsum += pow((testing_output - outputvalue), 2);
    }
    
    testrmse = sqrt((1.0 / (2.0 * numlines)) * testingsum);
    testresults[index] = testrmse;
}

void evaluate_net()
{
    // read in validation file
    ifstream infile(VALIDATION_FILE.c_str());
    if (!infile.is_open())
    {
        cerr << "Invalid validation file" << endl;
        exit(1);
    }
    
    vector<float> validation_inputs;
    validation_inputs.resize(NUM_INPUTS);
    float validation_output;
    float outputvalue;
    float validationsum = 0;
    int numlines = 0;
    float rmse;
    
    while (infile >> validation_inputs[0])
    {
        numlines++;
        
        for (int i = 1; i < NUM_INPUTS; i++)
        {
            infile >> validation_inputs[i];
        }
        infile >> validation_output;
        
        // compute outputs
        for(int i = 0; i < hiddenweights.size(); i++)
        {
            for(int j = 0; j < hiddenweights[i].size(); j++)
            {
                hiddenvalues[i][j] = 0.0;
                if(i == 0)
                {
                    // go through inputs
                    for (int k = 0; k < NUM_INPUTS; k++)
                    {
                        hiddenvalues[i][j] += hiddenweights[i][j][k] * validation_inputs[k];
                    }
                }
                else
                {
                    // go through previous hidden layer
                    for (int k = 0; k < hiddenweights[i][j].size(); k++)
                    {
                        hiddenvalues[i][j] += hiddenweights[i][j][k] * hiddenvalues[i-1][k];
                    }
                }
                
                hiddenvalues[i][j] += hiddenbiases[i][j];
                hiddenvalues[i][j] = 1.0 / (1.0 + exp(0.0 - hiddenvalues[i][j]));
            }
        }
        
        outputvalue = 0.0;
        for (int i = 0; i < hiddenvalues[hiddenvalues.size() - 1].size(); i++)
        {
            outputvalue += outputweights[i] * hiddenvalues[hiddenvalues.size() - 1][i];
        }
        outputvalue += outputbias;
        outputvalue = 1.0 / (1.0 + exp(0.0 - outputvalue));
        
        validationsum += pow((validation_output - outputvalue), 2);
    }
    
    validationrmse = sqrt((1.0 / (2.0 * numlines)) * validationsum);
}

void print_to_csv(ofstream &os)
{
    os << "Hidden Layers, " << HIDDEN_LAYERS << endl;
    os << "Num Neurons, ";
    for (int i = 0; i < HIDDEN_LAYERS; i++)
    {
        os << NUM_NEURONS[i];
        if (i < HIDDEN_LAYERS - 1)
            os << ", ";
    }
    os << endl;
    os << "Learning Rate, " << LEARNING_RATE << endl;
    os << "Training File, " << TRAINING_FILE << endl;
    os << "Testing File, " << TESTING_FILE << endl;
    os << "Validation File, " << VALIDATION_FILE << endl;
    os << "Num Epochs, " << NUM_EPOCHS << endl;
    os << "Num Inputs, " << NUM_INPUTS << endl << endl;
    os << "Validation RMSE, " << validationrmse << endl << endl;
    os << "Test Results" << endl << endl;
    for (int i = 0; i < NUM_EPOCHS; i++)
    {
        os << i << ", " << testresults[i] << endl;
    }
}

int main(int argc, char **argv)
{
    
    if (!(argc > 1))
    {
        cerr << "Output filename not specified" << endl;
        exit(1);
    }
    
    cout << "Enter number of hidden layers: ";
    cin >> HIDDEN_LAYERS;

    NUM_NEURONS.resize(HIDDEN_LAYERS);
    
    for (int i = 0; i < HIDDEN_LAYERS; i++)
    {
        cout << "Enter number of neurons in hidden layer " << i + 1 << ": ";
        cin >> NUM_NEURONS[i];
    }
    
    cout << "Enter learning rate: ";
    cin >> LEARNING_RATE;
    
    cout << "Enter training file name: ";
    cin >> TRAINING_FILE;

    cout << "Enter testing file name: ";
    cin >> TESTING_FILE;
    
    cout << "Enter validation file name: ";
    cin >> VALIDATION_FILE;
    
    cout << "Enter number of training epochs: ";
    cin >> NUM_EPOCHS;
    
    cout << "Enter the number of inputs: ";
    cin >> NUM_INPUTS;
    
    srand(time(NULL));
    
    init_weights();
    
    for (int i = 0; i < NUM_EPOCHS; i++)
    {
        train_net();
        test_net(i);
        cout << setw(8) << i + 1 << "/" << NUM_EPOCHS << endl;
    }
    
    evaluate_net();
    
        
    ofstream outfile(argv[1], ofstream::out);
    
    if (!outfile.is_open())
    {
        cerr << "Opening " << argv[1] << " failed" << endl;
        exit(1);
    }
    
    print_to_csv(outfile);
    
    outfile.close();
    
    return 0;
}
