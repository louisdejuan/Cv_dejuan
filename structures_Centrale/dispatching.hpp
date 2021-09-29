#ifndef _DISPATCHING__HPP_
#define _DISPATCHING__HPP_

#include "rnd.hpp"


/* Déclaration de la classe gérant le dispatching. */
class Dispatching
{
private:
    std::size_t _score = 100;
    bool _divergence = false;
    std::string _ordre = "";
    std::string _demande = "";
    double _objectif_MW = 0;

public:
    Dispatching() = default;

    const std::size_t& score() const {return _score;}
    const double& objectif_MW() const {return _objectif_MW;}
    const std::string& ordre() const {return _ordre;}

    /* Fonction d'actualisation du score et de l'ordre du QG. */
    const bool update(const double& MW);
};

#endif
