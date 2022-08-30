/********************************************************************************
* main.cpp: Implementering av en enkel maskininl�rningsmodell som baseras 
*           p� linj�r regression.
* 
*           I Windows, kompilera programkoden och skapa en k�rbar fil d�pt 
*           main (main.exe i Windows) via f�ljande kommando:
*           $ g++ main.cpp lin_reg.cpp -o main -Wall
*
*           F�r att k�ra programmet i Windows, skriv f�ljande kommando: 
*           $ main.exe (eller main.exe)
*  
*           F�r att k�ra programmet i Linux, skriv f�ljande kommando:
*           $ ./main
********************************************************************************/
#include "lin_reg.hpp"

/********************************************************************************
* main: Tr�nar en maskininl�rningsmodell baserad p� linj�r regression via 
*       tr�ningsdata best�ende av fem tr�ningsupps�ttningar. Modellen tr�nas 
*       under 100 epoker med en l�rhastighet p� 10 %. 
*
*       Efter tr�ningen �r slutf�rd sker prediktion f�r samtliga insignaler
*       mellan -5 och 5 med en stegringshastighet p� 0.5. Varje insignal
*       i detta intervall skrivs ut i terminalen tillsammans med predikterad
*       utsignal.
********************************************************************************/
int main(void)
{
   const std::vector<double> x = { -2, -1, 0, 1, 2 };
   const std::vector<double> yref = { -6, -4, -2, 0, 2 };

   lin_reg l1(x, yref);
   l1.train(100, 0.1);
   l1.predict_range(-5, 5, 0.5);
   return 0;
}