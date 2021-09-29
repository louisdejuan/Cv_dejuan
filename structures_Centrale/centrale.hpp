#ifndef _CENTRALE__HPP_
#define _CENTRALE__HPP_

#include <iostream>
#include <map>
#include <array>
#include <string>
#include <type_traits>

#include "rnd.hpp"



/* Énumération des états possibles des ouvriers. */
enum class Etat {Inactif, Actif, Malade};

/* Déclaration des fonctions de conversion d'un état d'ouvrier en chaîne de caractères. */
std::string toString(const Etat);
std::ostream& operator<<(std::ostream&, const Etat);

/* Énumération des organes de la centrale. */
enum class Organe
{
    None,
    Injecteur_acide_borique, Circuit_primaire, Pressuriseur, Pompe_circuit_primaire, Generateur_de_vapeur, Pompe_circuit_secondaire, Circuit_secondaire, Condenseur,
    Barre_de_graphite, Canaux, Piscine, Cuve, Echangeur_de_chaleur, Enceinte_de_confinement,
    First = Injecteur_acide_borique, Last = Enceinte_de_confinement,
    FirstReparable = First, LastReparable = Condenseur,
    FirstNonReparable = Barre_de_graphite, LastNonReparable = Last
};

/* Déclarations des surcharges des opérateurs sur les organes pour utiliser les itérateurs. */
Organe operator++(Organe&);
Organe operator*(Organe);
Organe begin(Organe);
Organe end(Organe);


/* Énumération des organes réparables de la centrale. */
enum class OrganeReparable : std::underlying_type<Organe>::type
{
    Injecteur_acide_borique, Circuit_primaire, Pressuriseur, Pompe_circuit_primaire, Generateur_de_vapeur, Pompe_circuit_secondaire, Circuit_secondaire, Condenseur,
    First = std::underlying_type<Organe>::type(Organe::FirstReparable),
    Last = std::underlying_type<Organe>::type(Organe::LastReparable)
};

/* Déclarations des surcharges des opérateurs sur les organes réparables pour utiliser les itérateurs. */
OrganeReparable operator++(OrganeReparable&);
OrganeReparable operator*(OrganeReparable);
OrganeReparable begin(OrganeReparable);
OrganeReparable end(OrganeReparable);


/* Énumération des organes non réparables de la centrale. */
enum class OrganeNonReparable : std::underlying_type<Organe>::type
{
    Barre_de_graphite, Canaux, Piscine, Cuve, Echangeur_de_chaleur, Enceinte_de_confinement,
    First = std::underlying_type<Organe>::type(Organe::FirstNonReparable),
    Last = std::underlying_type<Organe>::type(Organe::LastNonReparable)
};

/* Déclarations des surcharges des opérateurs sur les organes non réparables pour utiliser les itérateurs. */
OrganeNonReparable operator++(OrganeNonReparable&);
OrganeNonReparable operator*(OrganeNonReparable);
OrganeNonReparable begin(OrganeNonReparable);
OrganeNonReparable end(OrganeNonReparable);


/* Déclaration des fonctions de conversion d'un organe en chaîne de caractères. */
std::string toString(Organe);
std::ostream& operator<<(std::ostream&, Organe);

/* Énumération des états d'urgence concernant la centrale. */
enum class Urgence {DEFCOM_0, DEFCOM_1, DEFCOM_2, DEFCOM_3, DEFCOM_4};


/* Conteneurs pour les fonctions d'évaluation des états et des postes des ouvriers. */
typedef std::map<const Etat, std::size_t> map_etats_t;
typedef std::map<const Organe, std::size_t> map_postes_t;


/* Pair contenant le composant et l'alerte associée pour les alertes de sécurité. */
typedef std::pair<const Organe, const std::string> alerte_t;

/* Multimap contenant des alertes de sécurité. */
typedef std::multimap<const Organe, const std::string> map_alertes_t;


/* Conteneurs pour la fonction de réparabilité des organes de la centrale. */
typedef std::pair<const Organe, const bool> reparable_t;
typedef std::map<const Organe, const bool> map_reparables_t;


/* Classe représentant l'ensemble de la centrale. */
class Centrale
{
private:
    /* Variables non paramétrables de la centrale. */
    double _E_enceinte = 1.;
    double _P_enceinte = 1.;
    double _R_enceinte = 0.00002;
    double _E_centrale = 1.;
    double _MW = 0.;

    bool _arret_urgence = false;
    bool _evacuation_population = false;

    /* Classe représentant un ouvrier. */
    class Ouvrier
    {
    private:
        Etat _etat;
        Organe _poste;
        std::size_t _nb_jours_malade = 0;
        std::size_t _nb_jours_generateur_vapeur = 0;

    public:
        Ouvrier() : _etat(Etat::Inactif), _poste(Organe::None) {}

        const Etat etat() const {return _etat;}
        const Organe poste() const {return _poste;}

        /* Fonction d'affectation d'un ouvrier à un poste. */
        void affectation(const Organe tache);

        /* Fonction d'actualisation de l'état d'un ouvrier. */
        void update(Centrale centrale);
    };

    std::array<Ouvrier, 145> _ouvriers;

public:
    /* Composants de la centrale. */

    /* Déclaration de la classe "Reacteur". */
    class Reacteur
    {
    private:
        /* Variables paramétrables du réacteur. */
        double _Tx_Gr = 0.;
        double _Tx_B = 0.47;

        /* Variables non paramétrables du réacteur. */
        double _TGr_eff = 0.;
        double _TB_eff = 0.47;
        double _E_cuve = 1.;
        double _R_pisc;
        double _E_pisc = 1.;
        double _E_barre = 1.;
        double _E_canaux = 1.;
        double _E_bore = 1.;

    public:
        /* Constructeur d'une instance de réacteur. */
        Reacteur() : _R_pisc(100. + Rnd::random(45)) {}

        /* Getters des variables du réacteur. */
        const double& Tx_Gr() const {return _Tx_Gr;}
        const double& Tx_B() const {return _Tx_B;}
        const double& TGr_eff() const {return _TGr_eff;}
        const double& TB_eff() const {return _TB_eff;}
        const double& E_cuve() const {return _E_cuve;}
        const double& R_pisc() const {return _R_pisc;}
        const double& E_pisc() const {return _E_pisc;}
        const double& E_barre() const {return _E_barre;}
        const double& E_canaux() const {return _E_canaux;}
        const double& E_bore() const {return _E_bore;}

        /* Setters des variables paramétrables du réacteur. */
        const double& Tx_Gr(double);
        const double& Tx_B(double);

        /* Fonction d'actualisation des variables du réacteur. */
        void update(Centrale);

        /* Fonction de calcul des alertes de sécurité du reacteur. */
        map_alertes_t alertes(Centrale);

        /* Fonction de calcul des réparations des organes du réacteur. */
        void reparation(Centrale);

        /* Fonction de calcul des dommages causés au réacteur dus à un arrêt d'urgence. */
        void dommages_arret_urgence();

        /* Fonction de calcul des dommages d'une catastrophe causés au réacteur. */
        void catastrophe(const std::size_t, Centrale);

    } reacteur;

    /* Déclaration de la classe "Circuit_primaire". */
    class Circuit_primaire
    {
    private:
        /* Variables paramétrables du circuit primaire. */
        double _T_press = 25.;
        double _F_p1 = 0.;

        /* Variables non paramétrables du circuit primaire. */
        double _E_C1 = 1.;
        double _E_p1 = 1.;
        double _E_press = 1.;
        double _E_res = 1.;
        double _E_EC = 1.;
        double _Tpress_eff = 25.;
        double _P1 = 1.;
        double _D1 = 0.;
        double _I_T1 = 0.;
        double _T1 = 25.;
        double _R1 = 0.;

    public:
        /* Constructeur d'une instance de circuit primaire. */
        Circuit_primaire() = default;

        /* Getters des variables du circuit primaire. */
        const double& T_press() const {return _T_press;}
        const double& F_p1() const {return _F_p1;}
        const double& E_C1() const {return _E_C1;}
        const double& E_p1() const {return _E_p1;}
        const double& E_press() const {return _E_press;}
        const double& E_res() const {return _E_res;}
        const double& E_EC() const {return _E_EC;}
        const double& Tpress_eff() const {return _Tpress_eff;}
        const double& P1() const {return _P1;}
        const double& D1() const {return _D1;}
        const double& I_T1() const {return _I_T1;}
        const double& T1() const {return _T1;}
        const double& R1() const {return _R1;}

        /* Setters des variables paramétrables du circuit primaire. */
        const double& T_press(double);
        const double& F_p1(double);

        /* Fonction d'actualisation des variables du circuit primaire. */
        void update(Centrale);

        /* Fonction de calcul des alertes de sécurité du circuit primaire. */
        map_alertes_t alertes(Centrale);

        /* Fonction de calcul des réparations des organes du circuit primaire. */
        void reparation(Centrale);

        /* Fonction de calcul des dommages d'une catastrophe causés au circuit primaire. */
        void catastrophe(const std::size_t, Centrale);

    } circuit_primaire;

    /* Déclaration de la classe "Circuit_secondaire". */
    class Circuit_secondaire
    {
    private:
        /* Variables paramétrables du circuit secondaire. */
        double _F_p2 = 0.;
        double _F_cd = 0.;

        /* Variables non paramétrables du circuit secondaire. */
        double _E_C2 = 1.;
        double _E_p2 = 1.;
        double _E_vap = 1.;
        double _E_cd = 1.;
        double _T_vap = 25.;
        double _P_vap = 1.;
        double _D2 = 0.;
        double _D_cd = 0.;
        double _delta_ES = 0.;
        double _I_T2 = 0.;
        double _R2 = 0.;

    public:
        /* Constructeur d'une instance de circuit secondaire. */
        Circuit_secondaire() = default;

        /* Getters des variables du circuit secondaire. */
        const double& F_p2() const {return _F_p2;}
        const double& F_cd() const {return _F_cd;}
        const double& E_C2() const {return _E_C2;}
        const double& E_p2() const {return _E_p2;}
        const double& E_vap() const {return _E_vap;}
        const double& E_cd() const {return _E_cd;}
        const double& T_vap() const {return _T_vap;}
        const double& P_vap() const {return _P_vap;}
        const double& D2() const {return _D2;}
        const double& D_cd() const {return _D_cd;}
        const double& delta_ES() const {return _delta_ES;}
        const double& I_T2() const {return _I_T2;}
        const double& R2() const {return _R2;}

        /* Setters des variables paramétrables du circuit secondaire. */
        const double& F_p2(double);
        const double& F_cd(double);

        /* Fonction d'actualisation des variables du circuit secondaire. */
        void update(Centrale);

        /* Fonction de calcul des alertes de sécurité du circuit secondaire. */
        map_alertes_t alertes(Centrale);

        /* Fonction de calcul des réparations des organes du circuit secondaire. */
        void reparation(Centrale);

        /* Fonction de calcul des dommages d'une catastrophe causés au circuit secondaire. */
        void catastrophe(const std::size_t, Centrale);

    } circuit_secondaire;

    /* Déclaration de la classe représentant la population. */
    class Population
    {
    private:
        double _evac = 0;
        double _contamination = 0;
        double _R_air;
        double _R_eau;

    public:
        Population(Centrale centrale) :
            _R_air(centrale.E_enceinte() > 0.97 ? 0 : (1. - centrale.E_enceinte()) * centrale.R_enceinte() + (1. - centrale.circuit_secondaire.E_C2()) * 10),
            _R_eau((centrale.circuit_secondaire.E_cd() > 0.9 || centrale.circuit_secondaire.R2() < 2) ? 0 : (1. - centrale.circuit_secondaire.E_cd()) * centrale.circuit_secondaire.R2() / 100)
            {}

        /* Getters des variables de la population. */
        const double& evac() const {return _evac;}
        const double& contamination() const {return _contamination;}
        const double& R_air() const {return _R_air;}
        const double& R_eau() const {return _R_eau;}

        void update(Centrale centrale);
    } population;

    /* Constructeur d'une instance de centrale. */
    Centrale() : population(*this) {}

    /* Getters des variables de la centrale. */
    const double& E_enceinte() const {return _E_enceinte;}
    const double& P_enceinte() const {return _P_enceinte;}
    const double& R_enceinte() const {return _R_enceinte;}
    const double& E_centrale() const {return _E_centrale;}
    const double& MW() const {return _MW;}

    /* Fonction d'actualisation de toutes les variables de la centrale. */
    void update();

    /* Fonction de calcul des alertes de sécurité de la centrale. */
    map_alertes_t alertes();

    /* Fonction d'évaluation du nombre d'ouvriers dans chaque état (Inactif, Actif, ou Malade). */
    map_etats_t etats_ouvriers() const;

    /* Fonction d'évaluation du nombre d'ouvriers affectés à chaque organe de la centrale. */
    map_postes_t postes_ouvriers() const;

    /* Fonction d'évaluation de la réparabilité d'un organe de la centrale. */
    map_reparables_t organes_reparables() const;

    /* Fonction de calcul des réparations des organes de la centrale. */
    void reparation();

    /* Fonction d'affectation d'ouvriers à un poste. */
    const std::size_t affecter_ouvriers(const Organe, const std::size_t);

    /* Fonction de désaffectation des ouvriers d'un poste de réparation. */
    const std::size_t desaffecter_ouvriers(const Organe);

    /* Fonction d'alerte concernant l'état général de la centrale. */
    const Urgence etat();

    /* Getter de l'arrêt d'urgence de la centrale. */
    const bool& est_arrete() const {return _arret_urgence;}

    /* Fonction de tentative de lancement de la procédure d'arrêt d'urgence de la centrale. */
    const bool& arret_urgence();

    /* Getter de la demande d'évacuation de la population. */
    const bool& est_evacuee() const {return _evacuation_population;}

    /* Fonction de demande d'évacuation de la population. */
    const bool& evacuation_population() {_evacuation_population = true; return _evacuation_population;}

    /* Fonction de calcul des dommages d'une catastrophe potentielle causés à la centrale. */
    const std::string catastrophe();
};

#endif
