#include "tests.h"

int main(int argc, char **argv)
{
    if(std::string(argv[1]) == "testAdd")
    {
        testAdd(argc-2, argv+2);
    } 
    else if(std::string(argv[1]) == "testAddNPP")
    {
        testAddNPP();
    }
    else if(std::string(argv[1]) == "testAddC")
    {
        testAddC(argc - 2, argv + 2);
    }
    else if(std::string(argv[1]) == "testAddCNPP")
    {
        testAddCNPP();
    }
	else if (std::string(argv[1]) == "testAddCMulC")
	{
		testAddCMulC(argc - 2, argv + 2);
	}
	else if (std::string(argv[1]) == "testAddCMulCNPP")
	{
		testAddCMulCNPP();
	}
	else if (std::string(argv[1]) == "testMulC")
	{
		testMulC(argc - 2, argv + 2);
	}
	else if (std::string(argv[1]) == "testMulCNPP")
	{
		testMulCNPP();
	}



    return 0;
}