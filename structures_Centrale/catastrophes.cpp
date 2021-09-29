#include "rnd.hpp"
#include "centrale.hpp"



/* Fonction de calcul des dommages d'une catastrophe causés au réacteur. */
void Centrale::Reacteur::catastrophe(const std::size_t numero_catastrophe, Centrale centrale)
{
    switch(numero_catastrophe)
    {
       case 1: _E_pisc-=18;
               _E_cuve-=0.1+Rnd::random(0.1);
               _E_bore-=0.5+Rnd::random(0.25);
               _E_barre/=1.5;
               break;

       case 2: _E_pisc=0.5-Rnd::random(0.15);
               _E_cuve=0.55-Rnd::random(0.07);
               _E_barre=0.2+Rnd::random(0.02);
               _E_canaux/=1.5;
               _E_bore-=0.08+Rnd::random(0.1);
               break;

       case 3: _E_bore=Rnd::random(1.)*_E_bore;
               _E_barre-=0.2+Rnd::random(0.1);
               break;

       case 4: _E_pisc/=1.5;
               _E_cuve-=0.2;
               _E_barre/=1.2;
               _E_bore=_E_bore/1.4-Rnd::random(0.1);
               break;

       case 5: _E_pisc-=0.3+Rnd::random(0.3);
               _E_cuve=0.2+Rnd::random(0.3);
               double rdr1=Rnd::random(100);
               if(rdr1<=40)
                   _E_bore/=3.2;
               double rdr2=Rnd::random(100);
               if(rdr2<=30)
               {
                   _E_canaux/=1.5;
                   _E_barre/=2.4;
               }
   }
}


/* Fonction de calcul des dommages d'une catastrophe causés au circuit primaire. */
void Centrale::Circuit_primaire::catastrophe(const std::size_t numero_catastrophe, Centrale centrale)
{
    switch(numero_catastrophe)
    {
       case 1: _E_C1=0.3+Rnd::random(0.15);
               _E_press=0.65+Rnd::random(0.15);
               _E_p1=_E_C1+Rnd::random(0.03);
               _E_EC-=0.1+Rnd::random(0.15);
               _E_res=_E_press;
               break;

       case 2: _E_C1-=0.1+Rnd::random(0.13);
               _E_press-=0.3;
               _E_res-=0.4;
               break;

       case 3: _E_EC=0.5+Rnd::random(0.03);
               break;

       case 4: _E_EC-=0.03+Rnd::random(0.12);
               _E_C1-=0.2+Rnd::random(0.15);
               _E_p1=Rnd::random(1.)*_E_p1;
               break;

       case 5: _E_p1/=1.6;
               _E_C1-=0.2+Rnd::random(0.8);
               double rd1=Rnd::random(100);
               if(rd1<=70)
                   _E_C1/=1.4;
   }
}


/* Fonction de calcul des dommages d'une catastrophe causés au circuit secondaire. */
void Centrale::Circuit_secondaire::catastrophe(const std::size_t numero_catastrophe, Centrale centrale)
{
    switch(numero_catastrophe)
    {
        case 1: break;

        case 2: _E_vap-=0.03+Rnd::random(0.1);
                break;

        case 3: _E_vap=0.4+Rnd::random(0.1);
                _E_C2-=0.25+Rnd::random(0.03);
                _E_cd-=0.03;
                _E_p2-=0.04+Rnd::random(0.03);
                break;

        case 4: _E_vap-=0.14+Rnd::random(0.03);
                _E_p2/=1.2;
                _E_C2-=0.08+Rnd::random(0.1);
                _E_cd=_E_cd/2-Rnd::random(0.14);
                break;

        case 5: _E_cd/=2.6;
                _E_vap=0.1+Rnd::random(0.1);
                double rd2=Rnd::random(100);
                if(rd2 <= 50)
                {
                    _E_C2/=1.4;
                    _E_p2/=2;
                }
    }
}


/* Fonction de calcul des dommages d'une catastrophe causés à la centrale. */
const std::string Centrale::catastrophe()
{
    static std::size_t nb_tours_derniere_catastrophe = 10;

    if(MW() < 900)
    {
        nb_tours_derniere_catastrophe = 10;
    }
    else if(nb_tours_derniere_catastrophe++ == 10)
    {
        nb_tours_derniere_catastrophe = 1;

        if(Rnd::random(100) < 20)
        {
            const std::size_t numero_catastrophe = Rnd::randomInteger(1, 5);

            switch(numero_catastrophe)
            {
                case 1: break;
                case 2: break;
                case 3: _E_enceinte-=Rnd::random(0.03);
                break;

                case 4: _E_enceinte/=2.5;
                break;

                case 5: _E_enceinte/=2.6;
            }

            reacteur.catastrophe(numero_catastrophe, *this);
            circuit_primaire.catastrophe(numero_catastrophe, *this);
            circuit_secondaire.catastrophe(numero_catastrophe, *this);

            switch(numero_catastrophe)
            {
                case 1: return "Une catastrophe dans le circuit primaire ! La raison ? Ta gueule c'est magique ! ";
                case 2: return "La cuve et la piscine viennent de se dégrader ! Sortez les maillots de bain ! ";
                case 3: _E_enceinte-=Rnd::random(0.03);
                        return "Une fuite au niveau du générateur de vapeur ! Quelqu'un a une rustine ? ";

                case 4: _E_enceinte/=2.5;
                        return "Oh regardez, un avion... UN AVION ?!? ";

                case 5: _E_enceinte/=2.6;
                        return "Un tremblement de terre ! Ça s'est terminé comment au Japon déjà ? ";
            }
        }
    }

    return "Jusqu'ici tout va bien... Enfin, pour le moment... ";
}
