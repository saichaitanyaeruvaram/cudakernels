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
	else if (std::string(argv[1]) == "testBrightnessContrast_uv_int8")
	{
		testBrightnessContrast_uv_int8(argc - 2, argv + 2);
	}
	else if (std::string(argv[1]) == "testYUV420ToRGBNPP")
	{
		testYUV420ToRGBNPP();
	}
	else if (std::string(argv[1]) == "testYUV420ToRGB")
	{
		testYUV420ToRGB(argc - 2, argv + 2);
	}
	else if (std::string(argv[1]) == "testRGBToYUV420NPP")
	{
		testRGBToYUV420NPP();
	}
	else if (std::string(argv[1]) == "testRGBToYUV420")
	{
		testRGBToYUV420(argc - 2, argv + 2);
	}



    return 0;
}