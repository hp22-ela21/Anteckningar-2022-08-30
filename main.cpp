/********************************************************************************
* main.cpp: Implementering av en enkel maskininlärningsmodell som baseras 
*           på linjär regression.
* 
*           I Windows, kompilera programkoden och skapa en körbar fil döpt 
*           main (main.exe i Windows) via följande kommando:
*           $ g++ main.cpp lin_reg.cpp -o main -Wall
*
*           För att köra programmet i Windows, skriv följande kommando: 
*           $ main.exe (eller main.exe)
*  
*           För att köra programmet i Linux, skriv följande kommando:
*           $ ./main
********************************************************************************/
#include "lin_reg.hpp"

/********************************************************************************
* main: Tränar en maskininlärningsmodell baserad på linjär regression via 
*       träningsdata bestående av fem träningsuppsättningar. Modellen tränas 
*       under 100 epoker med en lärhastighet på 10 %. 
*
*       Efter träningen är slutförd sker prediktion för samtliga insignaler
*       mellan -5 och 5 med en stegringshastighet på 0.5. Varje insignal
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