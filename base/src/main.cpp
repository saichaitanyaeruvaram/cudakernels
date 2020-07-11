#include "common.h"

int main(int argc, char **argv)
{
    if(std::string(argv[1]) == "testAdd")
    {
        testAdd();
    } 
    else if(std::string(argv[1]) == "testAddNPP")
    {
        testAddNPP();
    }

    return 0;
}