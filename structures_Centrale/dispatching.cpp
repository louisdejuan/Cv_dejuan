#include "dispatching.hpp"



const bool Dispatching::update(const double& MW)
{
    if(_divergence)
    {
        static std::size_t nb_tours_arret = 0;

        if(MW == 0) ++nb_tours_arret;
        else nb_tours_arret = 0;

        if(_score == 0 || (nb_tours_arret >= 6 && Rnd::random(100) < 50))
            return true;

        static std::size_t nb_tours_ordre = 0;
        static std::size_t nb_tours_objectif_atteint = 0;

        if(nb_tours_ordre > 0 && std::fabs(_objectif_MW - MW) <= 10)
        {
            if(++nb_tours_objectif_atteint == 15)
            {
                nb_tours_ordre = 0;
                nb_tours_objectif_atteint = 0;
                _score += 10;
                _ordre = "Bien joué ! Mais c'est loin d'être terminé ! ";
                return false;
            }
        }

        if(nb_tours_ordre == 0)
        {
            if(Rnd::random(100) < 40)
            {
                _objectif_MW = Rnd::random(900, 1400);
                _demande = "Objectif : ";
            }
            else
                return false;
        }

        if(nb_tours_ordre < 8)
        {
            ++nb_tours_ordre;
        }
        else if(nb_tours_ordre < 16)
        {
            ++nb_tours_ordre;
            if(_score < 1)
            {
                _score = 0;
                return true;
            }
            _score -= 1;
        }
        else if(nb_tours_ordre == 16)
        {
            ++nb_tours_ordre;
            if(_score < 1)
            {
                _score = 0;
                return true;
            }
            _score -= 1;
            if(Rnd::random(100) < 20)
            {
                _objectif_MW = Rnd::random(100, 1400);
                _demande = "Nouvel objectif : ";
            }
        }
        else if(nb_tours_ordre < 50)
        {
            ++nb_tours_ordre;
            if(_score < 1)
            {
                _score = 0;
                return true;
            }
            _score -= 1;
        }
        else if(nb_tours_ordre < 65)
        {
            ++nb_tours_ordre;
            if(_score < 6)
            {
                _score = 0;
                return true;
            }
            _score -= 6;
            _demande = "Accélérez la procédure : ";
        }
        else
        {
            nb_tours_ordre = 0;
            if(_score < 3)
            {
                _score = 0;
                return true;
            }
            _score -= 3;
            _ordre = "Ordre annulé... Nous n'avions aucune attente, mais nous sommes tout de même déçus... ";
            return false;
        }

        if(!_demande.empty())
            _ordre = _demande + std::to_string((int)_objectif_MW) + " MW   (" + std::to_string(nb_tours_ordre) + "/65)";
    }
    else
    {
        if(MW == 0)
        {
            _ordre = "Mise en divergence de la centrale";
            _objectif_MW = MW;
        }
        else
            _divergence = true;
    }

    return false;
}
