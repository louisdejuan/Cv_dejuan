#include "centrale.hpp"



/* Fonctions de conversion d'un état d'ouvrier en chaîne de caractères. */
std::string toString(const Etat etat)
{
    switch(etat)
    {
        case Etat::Inactif : return "Inactif";
        case Etat::Actif : return "Actif";
        case Etat::Malade : return "Malade";
    }

    return "";
}

std::ostream& operator<<(std::ostream& out, const Etat etat)
{
    return out << toString(etat);
}


/* Surcharge des opérateurs sur les organes pour utiliser les itérateurs. */
Organe operator++(Organe& o) {return o = (Organe)(std::underlying_type<Organe>::type(o) + 1);}
Organe operator*(Organe o) {return o;}
Organe begin(Organe o) {return Organe::First;}
Organe end(Organe o) {Organe l = Organe::Last; return ++l;}

/* Surcharge des opérateurs sur les organes réparables pour utiliser les itérateurs. */
OrganeReparable operator++(OrganeReparable& oR) {return oR = (OrganeReparable)(std::underlying_type<OrganeReparable>::type(oR) + 1);}
OrganeReparable operator*(OrganeReparable oR) {return oR;}
OrganeReparable begin(OrganeReparable oR) {return OrganeReparable::First;}
OrganeReparable end(OrganeReparable oR) {OrganeReparable lR = OrganeReparable::Last; return ++lR;}

/* Surcharge des opérateurs sur les organes non réparables pour utiliser les itérateurs. */
OrganeNonReparable operator++(OrganeNonReparable& oNR) {return oNR = (OrganeNonReparable)(std::underlying_type<OrganeNonReparable>::type(oNR) + 1);}
OrganeNonReparable operator*(OrganeNonReparable oNR) {return oNR;}
OrganeNonReparable begin(OrganeNonReparable oNR) {return OrganeNonReparable::First;}
OrganeNonReparable end(OrganeNonReparable oNR) {OrganeNonReparable lNR = OrganeNonReparable::Last; return ++lNR;}


/* Fonctions de conversion d'un organe en chaîne de caractères. */
std::string toString(Organe organe)
{
    switch(organe)
    {
        case Organe::None : return "None";
        case Organe::Barre_de_graphite : return "Barre de graphite";
        case Organe::Canaux : return "Canaux";
        case Organe::Injecteur_acide_borique : return "Injecteur acide borique";
        case Organe::Piscine : return "Piscine";
        case Organe::Cuve : return "Cuve";
        case Organe::Circuit_primaire : return "Circuit primaire";
        case Organe::Pressuriseur : return "Pressuriseur";
        case Organe::Echangeur_de_chaleur : return "Echangeur de chaleur";
        case Organe::Pompe_circuit_primaire : return "Pompe circuit primaire";
        case Organe::Generateur_de_vapeur : return "Generateur de vapeur";
        case Organe::Pompe_circuit_secondaire : return "Pompe circuit secondaire";
        case Organe::Circuit_secondaire : return "Circuit secondaire";
        case Organe::Condenseur : return "Condenseur";
        case Organe::Enceinte_de_confinement : return "Enceinte de confinement";
    }
    return "";
}

std::ostream & operator<<(std::ostream & os, Organe organe)
{
    os << toString(organe);
    return os;
}



        /***************************************************/
        /* Définition des méthodes de la classe "Reacteur". */
        /***************************************************/

const double& Centrale::Reacteur::Tx_Gr(double val)
{
    if(val < (1 - _E_barre)) return _Tx_Gr = 1 -_E_barre;
    if(val > 1) return _Tx_Gr = 1;
    return _Tx_Gr = val;
}

const double& Centrale::Reacteur::Tx_B(double val)
{
    if(val < 0) return _Tx_B = 0;
    if(val > 0.5) return _Tx_B = 0.5;
    return _Tx_B = val;
}


void Centrale::Reacteur::update(Centrale centrale)
{
    if(centrale.est_arrete())
        _Tx_Gr = 0;

    if(fabs(_TGr_eff - _Tx_Gr) >= 0.01)
        _TGr_eff += 0.01 * (_TGr_eff < _Tx_Gr ? 1 : -1);

    if(fabs(_TB_eff - _Tx_B) >= 0.01)
        _TB_eff += 0.01 * (_TB_eff < _Tx_B ? 1 : -1);

    _R_pisc = (1 - _E_cuve) * centrale.circuit_primaire.R1() + 100 + Rnd::random(45);
}


map_alertes_t Centrale::Reacteur::alertes(Centrale centrale)
{
    map_alertes_t m_alertes_reacteur;

    if(420 <= centrale.circuit_primaire.T1() && 0.2 < _Tx_Gr)
    {
        m_alertes_reacteur.insert(alerte_t(Organe::Barre_de_graphite, "Dégradation possible des barres de graphite"));
        _E_barre -= Rnd::random(0.03);
    }

    if(50 <= centrale.circuit_primaire.T1() && 0.4 < _TGr_eff)
    {
        if(_E_cuve < 0.7)
        {
            m_alertes_reacteur.insert(alerte_t(Organe::Barre_de_graphite, "Risque important de dégradation des barres de graphite"));
            _E_barre -= (Rnd::random(100) < 40 ? Rnd::random(0.03) : 0);
        }
        if(_E_canaux < 0.6)
        {
            m_alertes_reacteur.insert(alerte_t(Organe::Barre_de_graphite, "Risque de dégradation des barres de graphite"));
            _E_barre -= (Rnd::random(100) < 30 ? Rnd::random(0.02) : 0);
        }
    }

    if(50 <= centrale.circuit_primaire.T1() && _E_cuve < 0.5)
    {
        m_alertes_reacteur.insert(alerte_t(Organe::Canaux, "Risque important de dégradation importante des canaux"));
        _E_canaux -= (Rnd::random(100) < 50 ? Rnd::random(0.05) : 0);

        if(centrale.circuit_primaire.E_C1() < 0.5)
        {
            m_alertes_reacteur.insert(alerte_t(Organe::Injecteur_acide_borique, "Risque important de dégradation de l'injecteur d'acide borique"));
            _E_bore -= (Rnd::random(100) < 50 ? Rnd::random(0.02) : 0);
        }
    }

    if(50 <= centrale.circuit_primaire.T1() && centrale.circuit_primaire.E_C1() < 0.2)
    {
        m_alertes_reacteur.insert(alerte_t(Organe::Piscine, "Risque très important de dégradation de la piscine"));
        _E_pisc -= (Rnd::random(100) < 80 ? Rnd::random(0.06) : 0);
    }

    if(50 <= centrale.circuit_primaire.T1())
    {
        if(centrale.circuit_primaire.E_C1() < 0.6)
        {
            m_alertes_reacteur.insert(alerte_t(Organe::Cuve, "Risque de dégradation de la cuve"));
            _E_cuve -= (Rnd::random(100) < 40 ? Rnd::random(0.02) : 0);
        }
        else if(centrale.circuit_primaire.E_C1() < 0.3)
        {
            m_alertes_reacteur.insert(alerte_t(Organe::Cuve, "Risque de forte dégradation de la cuve"));
            _E_cuve -= (Rnd::random(100) < 30 ? Rnd::random(0.03) : 0);
        }
        else if(centrale.circuit_primaire.E_C1() < 0.2)
        {
            m_alertes_reacteur.insert(alerte_t(Organe::Cuve, "Risque très important de très forte dégradation de la cuve"));
            _E_cuve -= (Rnd::random(100) < 80 ? Rnd::random(0.06) : 0);
        }
    }

    if(centrale.E_enceinte() == 0 && Rnd::random(100) < 65)
    {
        m_alertes_reacteur.insert(alerte_t(Organe::Cuve, "Détérioration importante de la cuve"));
        m_alertes_reacteur.insert(alerte_t(Organe::Piscine, "Détérioration importante de la piscine"));

        _E_cuve -= 0.5 + Rnd::random(0.1);
        _E_pisc -= 0.6 + Rnd::random(0.16);
    }

    return m_alertes_reacteur;
}


/* Fonction de calcul des réparations des organes du réacteur. */
void Centrale::Reacteur::reparation(Centrale centrale)
{
    map_postes_t m_postes_ouvriers = centrale.postes_ouvriers();

    if(m_postes_ouvriers[Organe::Injecteur_acide_borique] >= 5 && Rnd::random(100) < 30)
    {
        if(_E_bore < 0.96)
            _E_bore += Rnd::random(0.04);
    }

    if(0.96 <= _E_bore)
    {
        // for(std::size_t i = 0; i < 145; ++i)
        // {
        //     if(centrale._ouvriers[i].poste() == Organe::Injecteur_acide_borique)
        //     centrale._ouvriers[i].affectation(Organe::None);
        // }
        centrale.desaffecter_ouvriers(Organe::Injecteur_acide_borique);
    }
}

/* Fonction de calcul des dommages causés au réacteur dus à un arrêt d'urgence. */
void Centrale::Reacteur::dommages_arret_urgence()
{
    if(Rnd::random(100) < 80)
        _E_canaux -= 0.05 + Rnd::random(0.08);

    if(Rnd::random(100) < 30)
        _E_barre -= 0.02 + Rnd::random(0.08);

    if(Rnd::random(100) < 40)
        _E_cuve -= Rnd::random(0.08);
}



        /************************************************************/
        /* Définition des méthodes de la classe "Circuit_primaire". */
        /************************************************************/

const double& Centrale::Circuit_primaire::T_press(double val)
{
    if(val < 25) return _T_press = 25;
    if(val > 100 * (_E_press + _E_res)) return _T_press = 100 * (_E_press + _E_res);
    return _T_press = val;
}

const double& Centrale::Circuit_primaire::F_p1(double val)
{
    if(val < 0) return _F_p1 = 0;
    if(val > _E_p1) return _F_p1 = _E_p1;
    return _F_p1 = val;
}


void Centrale::Circuit_primaire::update(Centrale centrale)
{
    if(fabs(_Tpress_eff - _T_press) >= 1)
        _Tpress_eff += (_Tpress_eff < _T_press ? 1 : -1);

    _P1 = std::max(1., (_Tpress_eff - 25) / 3.24 + (_T1 - 100) / 83.3 * (_E_C1 + 0.1) * (_E_EC + 0.1));

    _D1 = _E_C1 * centrale.reacteur.E_cuve() * _F_p1 * 90;

    if(_E_EC < 50 && centrale.reacteur.TB_eff() < 25 && centrale.reacteur.TGr_eff() < 50)
        _I_T1 += Rnd::random(26);
    if(centrale.circuit_secondaire.T_vap() < _T1)
        _I_T1 += (centrale.circuit_secondaire.T_vap() - _T1) / 3;
    else _I_T1 = std::max(0., _I_T1 - Rnd::random(16));

    _T1 = std::max(25., (0.5 - centrale.reacteur.TB_eff()) / 0.5 * (645 * centrale.reacteur.TGr_eff() - 140 * _D1 / 90 + 2 * _P1) + 26 + _I_T1);

    _R1 = 98 * (_D1 + 1) * Rnd::random(90) + (0.5 - centrale.reacteur.TB_eff()) * (centrale.MW() + 0.1) * 6.54;
}


map_alertes_t Centrale::Circuit_primaire::alertes(Centrale centrale)
{
    map_alertes_t m_alertes_circuit_primaire;

    if(400 < _T1 && _T1 < 420)
    {
        m_alertes_circuit_primaire.insert(alerte_t(Organe::Circuit_primaire, "Température trop élevée dans le circuit primaire"));
    }

    if(420 <= _T1)
    {
        m_alertes_circuit_primaire.insert(alerte_t(Organe::Circuit_primaire, "Risque important de dégradation du circuit primaire dû à la température"));
        _E_C1 -= Rnd::random(0.02);

        m_alertes_circuit_primaire.insert(alerte_t(Organe::Pressuriseur, "Dégradation possible du pressuriseur"));
        if(Rnd::random(100) < 30)
        {
            _E_press -= Rnd::random(0.02);
            _E_res -= Rnd::random(0.02);
        }
    }

    if(_P1 * 40 <= _T1)
    {
        m_alertes_circuit_primaire.insert(alerte_t(Organe::Circuit_primaire, "Risque important de dégradation du circuit primaire dû à la formation de vapeur dans le circuit primaire"));
        _E_C1 -= Rnd::random(0.03);

        m_alertes_circuit_primaire.insert(alerte_t(Organe::Pressuriseur, "Risque important de dégradation du pressuriseur dû à la formation de vapeur dans le circuit primaire"));
        _E_press -= Rnd::random(0.02);

        m_alertes_circuit_primaire.insert(alerte_t(Organe::Pressuriseur, "Risque de dégradation de la résistance du pressuriseur dû à la formation de vapeur dans le circuit primaire"));
        _E_res -= (Rnd::random(100) < 30 ? Rnd::random(0.02) : 0);
    }

    if(50 <= _T1)
    {
        if(_E_C1 < 0.5)
        {
            m_alertes_circuit_primaire.insert(alerte_t(Organe::Echangeur_de_chaleur, "Risque important de dégradation de l'échangeur de chaleur"));
            _E_EC -= (Rnd::random(100) < 40 ? Rnd::random(0.04) : 0);
        }

        if(_E_C1 < 0.58)
        {
            m_alertes_circuit_primaire.insert(alerte_t(Organe::Pompe_circuit_primaire, "Risque important de dégradation de la pompe du circuit primaire"));
            _E_p1 -= (Rnd::random(100) < 50 ? Rnd::random(0.03) : 0);
        }

        if(_E_press < 0.5)
        {
            m_alertes_circuit_primaire.insert(alerte_t(Organe::Pressuriseur, "Risque très important de dégradation de la résistance du pressuriseur"));
            _E_res -= (Rnd::random(100) < 80 ? Rnd::random(0.03) : 0);
        }

        if(10 < _P1)
        {
            m_alertes_circuit_primaire.insert(alerte_t(Organe::Circuit_primaire, "Risque faible de dégradation du circuit primaire dû à une pression trop forte"));
            _E_C1 -= (Rnd::random(100) < 20 ? Rnd::random(0.02) : 0);
        }

        if(_E_EC < 0.6)
        {
            m_alertes_circuit_primaire.insert(alerte_t(Organe::Circuit_primaire, "Risque faible de dégradation du circuit primaire"));
            _E_C1 -= (Rnd::random(100) < 30 ? Rnd::random(0.015) : 0);
        }
    }

    if(centrale.E_enceinte() == 0 && Rnd::random(100) < 80)
    {
        m_alertes_circuit_primaire.insert(alerte_t(Organe::Circuit_primaire, "Détérioration importante du circuit primaire"));
        m_alertes_circuit_primaire.insert(alerte_t(Organe::Pressuriseur, "Détérioration importante du pressuriseur"));

        if(_E_press < 0.9)
            _E_res -= (Rnd::random(100) < 80 ? Rnd::random(0.1) : 0);

        _E_C1 -= 0.1 + Rnd::random(0.1);
        _E_press -= 0.1;
    }

    return m_alertes_circuit_primaire;
}


/* Fonction de calcul des réparations des organes du circuit primaire. */
void Centrale::Circuit_primaire::reparation(Centrale centrale)
{
    map_postes_t m_postes_ouvriers = centrale.postes_ouvriers();

    if(m_postes_ouvriers[Organe::Pompe_circuit_primaire] >= 8 && Rnd::random(100) < 30)
    {
        if(_E_p1 <= 0.97)
            _E_p1 += Rnd::random(0.03);
        else
            _E_p1 = 1;
    }

    if(_E_p1 == 1)
    {
        // for(std::size_t i = 0; i < 145; ++i)
        // {
        //     if(centrale._ouvriers[i].poste() == Organe::Pompe_circuit_primaire)
        //     centrale._ouvriers[i].affectation(Organe::None);
        // }
        centrale.desaffecter_ouvriers(Organe::Pompe_circuit_primaire);
    }

    if(m_postes_ouvriers[Organe::Circuit_primaire] >= 45 && Rnd::random(100) < 30)
    {
        if(_E_C1 < 0.8)
            _E_C1 += Rnd::random(0.02);
    }

    if(0.8 <= _E_C1)
    {
        // for(std::size_t i = 0; i < 145; ++i)
        // {
        //     if(centrale._ouvriers[i].poste() == Organe::Circuit_primaire)
        //         centrale._ouvriers[i].affectation(Organe::None);
        // }
        centrale.desaffecter_ouvriers(Organe::Circuit_primaire);
    }

    if(m_postes_ouvriers[Organe::Pressuriseur] >= 10 && Rnd::random(100) < 30)
    {
        if(_E_press < 0.97)
        {
            _E_press += Rnd::random(0.03);
            _E_res += Rnd::random(0.04);
            _E_res = std::max(_E_press, _E_res);
        }
        else
        {
            _E_press = 1;
            _E_res = 1;
        }
    }

    if(_E_press == 1)
    {
        // for(std::size_t i = 0; i < 145; ++i)
        // {
        //     if(centrale._ouvriers[i].poste() == Organe::Pressuriseur)
        //         centrale._ouvriers[i].affectation(Organe::None);
        // }
        centrale.desaffecter_ouvriers(Organe::Pressuriseur);
    }
}



        /**************************************************************/
        /* Définition des méthodes de la classe "Circuit_secondaire". */
        /**************************************************************/

const double& Centrale::Circuit_secondaire::F_p2(double val)
{
    if(val < 0) return _F_p2 = 0;
    if(val > _E_p2) return _F_p2 = _E_p2;
    return _F_p2 = val;
}

const double& Centrale::Circuit_secondaire::F_cd(double val)
{
    if(val < 0) return _F_cd = 0;
    if(val > _E_cd) return _F_cd = _E_cd;
    return _F_cd = val;
}


void Centrale::Circuit_secondaire::update(Centrale centrale)
{
    _T_vap = std::max(99., centrale.circuit_primaire.E_EC() * (centrale.circuit_primaire.T1() / 1.51) + 26 + (_I_T2 > 3 ? _I_T2 : 0));

    _P_vap = (_T_vap < 120 ? 1. : std::max(1., (_E_C2 + 0.1) * _E_vap * (_F_p2 / 50 + (_T_vap - 135) / 10)));

    double Z = _E_C2 * (_E_cd + 0.1) * _E_vap * _F_p2 * 0.85;
    _D2 = (_T_vap > 3 && Z < 3 ? _E_C2 * (_E_cd + 0.1) * _E_vap * 1.3 : Z);

    _D_cd = _E_cd * _F_cd * 150;

    _delta_ES = _D_cd / 7.5;

    if(_F_p2 < 0.55 + Rnd::random(0.15) && centrale.circuit_primaire.T1() > 200)
        _I_T2 += Rnd::random(centrale.circuit_primaire.T1() / 100);
    else if(_I_T2 > 0)
        _I_T2 = std::max(0., _I_T2 - Rnd::random(3));

    _R2 = std::max(_R2, (1. - centrale.circuit_primaire.E_EC()) * centrale.circuit_primaire.R1());

    if(centrale.E_enceinte() == 0)
        _E_vap -= 0.1 + Rnd::random(0.1);
}


map_alertes_t Centrale::Circuit_secondaire::alertes(Centrale centrale)
{
    map_alertes_t m_alertes_circuit_secondaire;

    if(50 <= centrale.circuit_primaire.T1())
    {
        if(centrale.circuit_primaire.E_C1() < 0.5)
        {
            m_alertes_circuit_secondaire.insert(alerte_t(Organe::Generateur_de_vapeur, "Risque de dégradation du générateur de vapeur"));
            _E_vap -= (Rnd::random(100) < 20 ? Rnd::random(0.02) : 0);
        }

        if(centrale.circuit_primaire.E_EC() < 0.6 && 3 < centrale.circuit_primaire.P1())
        {
            m_alertes_circuit_secondaire.insert(alerte_t(Organe::Generateur_de_vapeur, "Risque important de dégradation du générateur de vapeur"));
            _E_vap -= (Rnd::random(100) < 50 ? Rnd::random(0.02) : 0);
        }
    }

    if(120 <= _T_vap)
    {
        if(_E_C2 < 0.6)
        {
            m_alertes_circuit_secondaire.insert(alerte_t(Organe::Pompe_circuit_secondaire, "Risque important de dégradation de la pompe du circuit secondaire"));
            _E_p2 -= (Rnd::random(100) < 30 ? Rnd::random(0.02) : 0);
        }

        if(_E_C2 < 0.4)
        {
            m_alertes_circuit_secondaire.insert(alerte_t(Organe::Generateur_de_vapeur, "Risque important de dégradation du générateur de vapeur"));
            _E_vap -= (Rnd::random(100) < 40 ? Rnd::random(0.02) : 0);
        }

        if(centrale.circuit_primaire.E_EC() < 0.7)
        {
            m_alertes_circuit_secondaire.insert(alerte_t(Organe::Echangeur_de_chaleur, "Problème d'échange de chaleur entre le circuit primaire et le circuit secondaire"));
        }

        if(centrale.circuit_primaire.E_EC() < 0.4)
        {
            m_alertes_circuit_secondaire.insert(alerte_t(Organe::Echangeur_de_chaleur, "Risque important de dégradation de l'échangeur de chaleur"));
            _E_vap -= (Rnd::random(100) < 25 ? Rnd::random(0.03) : 0);
        }

        if(130 <= _T_vap)
        {
            m_alertes_circuit_secondaire.insert(alerte_t(Organe::Circuit_secondaire, "Risque important de dégradation du circuit secondaire à cause de la température"));
            _E_C2 -= (Rnd::random(100) < 25 ? Rnd::random(0.04) : 0);

            if(_delta_ES < 11)
            {
                m_alertes_circuit_secondaire.insert(alerte_t(Organe::Circuit_secondaire, "Risque possible de dégradation du circuit secondaire dû au refroidissement"));
            }

            if(_T_vap != 130 && _E_C2 < 0.5)
            {
                m_alertes_circuit_secondaire.insert(alerte_t(Organe::Circuit_secondaire, "Risque important de dégradation du circuit secondaire"));
                _E_C2 -= (Rnd::random(100) < 30 ? Rnd::random(0.015) : 0);
            }
        }
    }

    if(19 < _delta_ES)
    {
        m_alertes_circuit_secondaire.insert(alerte_t(Organe::Condenseur, "Dégradation potentielle du condenseur à cause de la différence de température"));
        _E_cd -= (Rnd::random(100) < 40 ? Rnd::random(0.03) : 0);
    }

    if(centrale.E_enceinte() == 0 && _E_vap < 0.4)
    {
        m_alertes_circuit_secondaire.insert(alerte_t(Organe::Generateur_de_vapeur, "Détérioration importante du générateur de vapeur"));
        _E_vap -= 0.1 + Rnd::random(0.1);
    }

    return m_alertes_circuit_secondaire;
}


void Centrale::Circuit_secondaire::reparation(Centrale centrale)
{
    map_postes_t m_postes_ouvriers = centrale.postes_ouvriers();

    if(m_postes_ouvriers[Organe::Pompe_circuit_secondaire] >= 8 && Rnd::random(100) < 30)
    {
        if(_E_p2 <= 0.97)
            _E_p2 += Rnd::random(0.03);
        else
            _E_p2 = 1;
    }

    if(_E_p2 == 1)
    {
        // for(std::size_t i = 0; i < 145; ++i)
        // {
        //     if(centrale._ouvriers[i].poste() == Organe::Pompe_circuit_secondaire)
        //     centrale._ouvriers[i].affectation(Organe::None);
        // }
        centrale.desaffecter_ouvriers(Organe::Pompe_circuit_secondaire);
    }

    if(m_postes_ouvriers[Organe::Condenseur] >= 20 && Rnd::random(100) < 30)
    {
        if(_E_cd <= 0.98)
            _E_cd += Rnd::random(0.02);
        else
            _E_cd = 1;
    }

    if(_E_cd == 1)
    {
        // for(std::size_t i = 0; i < 145; ++i)
        // {
        //     if(centrale._ouvriers[i].poste() == Organe::Condenseur)
        //     centrale._ouvriers[i].affectation(Organe::None);
        // }
        centrale.desaffecter_ouvriers(Organe::Condenseur);
    }

    if(m_postes_ouvriers[Organe::Generateur_de_vapeur] >= 60 && Rnd::random(100) < 30)
    {
        if(_E_vap < 0.89)
            _E_vap += Rnd::random(0.05);
    }

    if(0.89 <= _E_vap)
    {
        // for(std::size_t i = 0; i < 145; ++i)
        // {
        //     if(centrale._ouvriers[i].poste() == Organe::Generateur_de_vapeur)
        //         centrale._ouvriers[i].affectation(Organe::None);
        // }
        centrale.desaffecter_ouvriers(Organe::Generateur_de_vapeur);
    }

    if(m_postes_ouvriers[Organe::Circuit_secondaire] >= 20 && Rnd::random(100) < 30)
    {
        if(_E_C2 < 0.78)
            _E_C2 += Rnd::random(0.02);
    }

    if(0.78 <= _E_C2)
    {
        // for(std::size_t i = 0; i < 145; ++i)
        // {
        //     if(centrale._ouvriers[i].poste() == Organe::Circuit_secondaire)
        //         centrale._ouvriers[i].affectation(Organe::None);
        // }
        centrale.desaffecter_ouvriers(Organe::Circuit_secondaire);
    }
}



        /****************************************************/
        /* Définition des méthodes de la classe "Centrale". */
        /****************************************************/

void Centrale::update()
{
    reacteur.update(*this);

    circuit_primaire.update(*this);

    circuit_secondaire.update(*this);

    if((circuit_primaire.P1() > 8 && reacteur.E_cuve() < 1 && reacteur.E_pisc() < 1) || (reacteur.E_cuve() < 0.3 && reacteur.E_pisc() < 0.4))
        _P_enceinte += (2. - reacteur.E_cuve() - reacteur.E_pisc()) / 23;
    if((circuit_primaire.E_C1() < 0.6 && circuit_primaire.P1() > 12) || circuit_primaire.E_C1() < 0.2)
        _P_enceinte += (1. - circuit_primaire.E_C1()) / 30;
    if(circuit_secondaire.E_vap() < 0.9 && circuit_secondaire.P_vap() > 2)
        _P_enceinte += (1. - circuit_secondaire.E_vap()) / 20;
    _P_enceinte = std::min(5., _P_enceinte);
    if(_E_enceinte > 0.5 && _E_enceinte < 0.6)
        _P_enceinte -= (1. - _E_enceinte) / 2 * _P_enceinte;
    if(Rnd::random(100) < 30)
        _P_enceinte -= 0.13;

    _R_enceinte = Rnd::random(1. / 55) + 0.00002 + (1. - circuit_primaire.E_C1()) * circuit_primaire.R1() / 98.98 + (1. - circuit_primaire.E_press()) * 10;
    if(reacteur.E_pisc() < 0.55 && reacteur.R_pisc() > 3000)
        _R_enceinte *= 1.25;
    if(_E_enceinte < 0.9)
        _R_enceinte /= 1.3;
    if(_E_enceinte < 0.1)
        _R_enceinte /= 1.5;

    _E_centrale = (reacteur.E_canaux() + 2 * reacteur.E_barre() + 8 * reacteur.E_cuve() + 3 * reacteur.E_pisc() + circuit_primaire.E_p1() + circuit_secondaire.E_p2() + 5 * circuit_primaire.E_EC() + 4 * circuit_secondaire.E_vap() + circuit_primaire.E_press() + circuit_primaire.E_res() + 4 * _E_enceinte + 8 * circuit_primaire.E_C1() + 3 * circuit_secondaire.E_C2() + reacteur.E_bore() + circuit_secondaire.E_cd()) / 44;

    if(circuit_secondaire.T_vap() < 120 || circuit_secondaire.E_C2() < 0.22)
        _MW = 0;
    else
    {
        if(circuit_secondaire.T_vap() < 3000)
            _MW = std::max(0., 5.787 * (circuit_secondaire.T_vap() - 120.) + 28.118 * (circuit_secondaire.P_vap() - 1.) + circuit_primaire.P1());
        else _MW = 30000;
        if(circuit_secondaire.E_C2() < 0.6)
            _MW *= circuit_secondaire.E_C2();
        if(_MW > 1400 && _MW < 1412)
            _MW = 1400;
    }

    population.update(*this);
}


map_alertes_t Centrale::alertes()
{
    map_alertes_t m_alertes_reacteur = reacteur.alertes(*this);

    map_alertes_t m_alertes_circuit_primaire = circuit_primaire.alertes(*this);

    map_alertes_t m_alertes_circuit_secondaire = circuit_secondaire.alertes(*this);

    map_alertes_t m_alertes_centrale;

    if(2 < _P_enceinte && _P_enceinte < 3.4)
    {
        m_alertes_centrale.insert(alerte_t(Organe::Enceinte_de_confinement, "Dégradation potentielle de l'enceinte de confinement dû à la pression"));

        if(0.5 < _E_enceinte)
        {
            _E_enceinte -= (Rnd::random(100) < 30 ? Rnd::random(0.02) : 0);
        }
    }
    else if(3.4 <= _P_enceinte && _P_enceinte < 4.5)
    {
        m_alertes_centrale.insert(alerte_t(Organe::Enceinte_de_confinement, "Risque important de dégradation de l'enceinte de confinement dû à la pression"));

        if(0.25 < _E_enceinte)
        {
            _E_enceinte -= (Rnd::random(100) < 60 ? Rnd::random(0.06) : 0);
        }
    }
    else if(4.5 <= _P_enceinte || _E_enceinte == 0)
    {
        m_alertes_centrale.insert(alerte_t(Organe::Enceinte_de_confinement, "Destruction totale de l'enceinte de confinement"));
        _E_enceinte = 0;
    }

    for(auto itr : m_alertes_reacteur)
        m_alertes_centrale.insert(itr);

    for(auto itr : m_alertes_circuit_primaire)
        m_alertes_centrale.insert(itr);

    for(auto itr : m_alertes_circuit_secondaire)
        m_alertes_centrale.insert(itr);

    return m_alertes_centrale;
}


/* Fonction d'évaluation du nombre d'ouvriers dans chaque état (Inactif, Actif, ou Malade). */
map_etats_t Centrale::etats_ouvriers() const
{
    map_etats_t m_etats_ouvriers;
    for(const auto& ouvrier : _ouvriers)
    {
        m_etats_ouvriers.try_emplace(ouvrier.etat(), 0);
        m_etats_ouvriers[ouvrier.etat()] += 1;
    }

    return m_etats_ouvriers;
}


/* Fonction d'évaluation du nombre d'ouvriers affectés à chaque organe de la centrale. */
map_postes_t Centrale::postes_ouvriers() const
{
    map_postes_t m_postes_ouvriers;
    for(const auto& ouvrier : _ouvriers)
    {
        m_postes_ouvriers.try_emplace(ouvrier.poste(), 0);
        m_postes_ouvriers[ouvrier.poste()] += 1;
    }

    return m_postes_ouvriers;
}


/* Fonction d'évaluation de la réparabilité d'un organe de la centrale. */
map_reparables_t Centrale::organes_reparables() const
{
    map_reparables_t m_organes_reparables;

    for(const auto& itr : OrganeNonReparable())
        m_organes_reparables.emplace(reparable_t((Organe)itr, false));

    /* Test de réparabilité du circuit primaire. */
    if(circuit_primaire.E_C1() < 0.2)
        m_organes_reparables.emplace(reparable_t(Organe::Circuit_primaire, false));

    /* Test de reparabilité du "chantier" (première condition) mdr... */
    if(R_enceinte() > 12 || P_enceinte() > 3)
        for(const auto& itr : OrganeReparable())
            m_organes_reparables.emplace(reparable_t((Organe)itr, false));

    /* Test de reparabilité du "chantier" (deuxième condition) mdr... */
    if(circuit_secondaire.T_vap()> 300 || MW() > 1000)
        for(const auto& itr : OrganeReparable())
            m_organes_reparables.emplace(reparable_t((Organe)itr, false));

    for(const auto& itr : OrganeReparable())
        m_organes_reparables.emplace(reparable_t((Organe)itr, true));

    return m_organes_reparables;
}


/* Fonction de calcul des réparations des organes de la centrale. */
void Centrale::reparation()
{
    reacteur.reparation(*this);
    circuit_primaire.reparation(*this);
    circuit_secondaire.reparation(*this);

    for(auto& itr : _ouvriers)
        itr.update(*this);
}


/* Fonction d'affectation d'ouvriers à un poste. */
const std::size_t Centrale::affecter_ouvriers(const Organe poste, const std::size_t nb_ouvriers)
{
    if(poste == Organe::None)
        return 0;

    if(etats_ouvriers()[Etat::Inactif] < nb_ouvriers)
        return 0;

    if(poste == Organe::Circuit_primaire && circuit_primaire.E_C1() < 0.2)
        return 0;

    if(R_enceinte() > 12 || P_enceinte() > 3 || circuit_secondaire.T_vap() > 300 || MW() > 1000)
        return 0;

    for(std::size_t i = 0; i < nb_ouvriers; ++i)
    {
        for(auto& itr : _ouvriers)
        {
            if(itr.etat() == Etat::Inactif)
            {
                itr.affectation(poste);
                break;
            }
        }
    }

    return nb_ouvriers;
}


/* Fonction de désaffectation des ouvriers d'un poste de réparation. */
const std::size_t Centrale::desaffecter_ouvriers(const Organe poste)
{
    if(poste == Organe::None)
        return 0;

    std::size_t nb_ouvriers = 0;

    for(auto& itr : _ouvriers)
    {
        if(itr.poste() == poste)
        {
            itr.affectation(Organe::None);
            ++nb_ouvriers;
        }
    }

    return nb_ouvriers;
}


/* Fonction d'alerte concernant l'état général de la centrale. */
const Urgence Centrale::etat()
{
    if(_E_centrale < 0.22)
        return Urgence::DEFCOM_4;

    if(_E_centrale < 0.3)
        return Urgence::DEFCOM_3;

    if(_E_centrale < 0.5)
        return Urgence::DEFCOM_2;

    if(_E_centrale < 0.6)
        return Urgence::DEFCOM_1;

    return Urgence::DEFCOM_0;
}


/* Fonction de tentative de lancement de la procédure d'arrêt d'urgence de la centrale. */
const bool& Centrale::arret_urgence()
{
    if(!est_arrete())
    {
        if(Rnd::random(1.) < 0.7 * reacteur.E_cuve() * reacteur.E_canaux() * reacteur.E_barre() + 0.15)
            _arret_urgence = true;

        else
            reacteur.dommages_arret_urgence();
    }

    return est_arrete();
}



/*************************************************/
/* Définition des méthodes de la classe Ouvrier. */
/*************************************************/

/* Définition de la fonction d'affectation d'une tâche à un ouvrier. */
void Centrale::Ouvrier::affectation(const Organe tache)
{
    _poste = tache;
    if(tache == Organe::None)
        _etat = Etat::Inactif;
    else
        _etat = Etat::Actif;
}


/* Définition de la fonction d'actualisation de l'état d'un ouvrier. */
void Centrale::Ouvrier::update(Centrale centrale)
{
    if(_etat == Etat::Malade && (++_nb_jours_malade) == 8)
    {
        _nb_jours_malade = 0;

        if(Rnd::random(100) < 40)
            _etat = Etat::Inactif;
    }

    if(_etat == Etat::Actif && (12 < centrale.R_enceinte() || 3 < centrale.P_enceinte()))
    {
        _poste = Organe::None;
        _etat = Etat::Malade;
        _nb_jours_malade = 0;
    }

    else if(_etat == Etat::Actif && (300 < centrale.circuit_secondaire.T_vap() || 1000 < centrale.MW()))
    {
        _poste = Organe::None;
        _etat = Etat::Malade;
        _nb_jours_malade = 0;
    }

    else if(_poste == Organe::Circuit_primaire && centrale.circuit_primaire.E_C1() < 0.2)
    {
        _poste = Organe::None;
        _etat = Etat::Malade;
        _nb_jours_malade = 0;
    }

    else if(_poste == Organe::Generateur_de_vapeur && (++_nb_jours_generateur_vapeur) == 5)
    {
        _nb_jours_generateur_vapeur = 0;

        if(Rnd::random(100) < 10)
        {
            _poste = Organe::None;
            _etat = Etat::Malade;
            _nb_jours_malade = 0;
        }
    }
}



        /******************************************************/
        /* Définition des méthodes de la classe "Population". */
        /******************************************************/

void Centrale::Population::update(Centrale centrale)
{
    _R_air = (centrale.E_enceinte() > 0.97 ? 0 : (1. - centrale.E_enceinte()) * centrale.R_enceinte() + (1. - centrale.circuit_secondaire.E_C2()) * 10);
    _R_eau = ((centrale.circuit_secondaire.E_cd() > 0.9 || centrale.circuit_secondaire.R2() < 2) ? 0 : (1. - centrale.circuit_secondaire.E_cd()) * centrale.circuit_secondaire.R2() / 100);

    if(_evac < 0.9)
    {
        if(centrale.est_evacuee() && Rnd::random(100) < 40)
            _evac += 0.10111111111;         // TOUJOURS PLUS DE 1111111111 !!! 

        if(centrale.R_enceinte() < 0.1)
            _contamination -= 5 * (_R_air > 6) - 10 * (_R_air > 12) - 5 * (_R_eau > 1) - 8 * (_R_eau > 12);

        _contamination += (_R_air > 12 ? 4 + Rnd::random(15) : 0);
        _contamination += (_R_air > 20 ? 12 + Rnd::random(20) : 0);
        _contamination += (_R_eau > 12 ? 5 + Rnd::random(20) : 0);
    }
    else
        _evac = 1;
}
