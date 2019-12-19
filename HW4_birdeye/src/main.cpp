#include "hw4.h"

int main(int argc, char** argv)
{
    switch (argc)
    {
    case 8:
        e18_1(argc, argv);
        break;
    case 9:
        e18_2(argc, argv);
        break;
    default:
        return -1;
    }
    e19_1(argc, argv);
}