/********************************************************************************
* lin_reg.hpp: Innehåller funktionalitet för enkel implementering av
*              maskininlärningsmodeller baserade på linjär regression via 
*              strukten lin_reg.
********************************************************************************/
#ifndef LIN_REG_HPP_
#define LIN_REG_HPP_

/* Inkluderingsdirektiv: */
#include <iostream>
#include <vector>

/********************************************************************************
* lin_reg: Strukt för implementering av regressionsmodell som baseras på
*          linjär regression. Modellens parametrar tilldelas randomiserade
*          startvärden mellan 0.0 - 1.0. Träningsdata passeras via referenser 
*          till vektorer innehållande träningsuppsättningarnas in- och utdata. 
********************************************************************************/
struct lin_reg
{
   /* Medlemmar: */
   std::vector<double> x;                /* Indata för träningsuppsättningar. */
   std::vector<double> yref;             /* Utdata för träningsuppsättningar. */
   std::vector<std::size_t> train_order; /* Lagrar ordningsföljd vid träning via index. */
   double bias = get_random();           /* Bias/vilovärde (m-värde). */
   double weight = get_random();         /* Vikt/lutning (k-värde). */

   /* Medlemsfunktioner: */
   lin_reg(void) { }
   lin_reg(const std::vector<double>& x,
           const std::vector<double>& yref) { this->set_training_data(x, yref); }
   std::size_t num_sets(void) { return this->train_order.size(); }
   void set_training_data(const std::vector<double>& x,
                          const std::vector<double>& yref);
   void train(const std::size_t num_epochs,
              const double learning_rate);
   double predict(const double input) { return this->weight * input + this->bias; }
   void predict_range(const double min, 
                      const double max,
                      const double step = 1.0,
                      std::ostream& ostream = std::cout);
private:
   double get_random(void) { return std::rand() / static_cast<double>(RAND_MAX); }
   void shuffle(void);
   void optimize(const double input, 
                  const double reference, 
                  const double learning_rate);
};

#endif /* LIN_REG_HPP_ */