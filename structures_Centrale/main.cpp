#include <iostream>

#include "rnd.hpp"
#include "centrale.hpp"
#include "dispatching.hpp"

#include <string>
#include <map>

/*
Update états ouvriers
Affecter postes ouvriers
Update réparations organes
Update centrale
Afficher alertes
*/

template <std::size_t N = 10>
class test
{
public:
    test() {std::cout << N << std::endl;}
};

int main()
{
    std::cout << Rnd::randomInteger(10) << std::endl;
    // for(const Organe& itr : Organe())
    // {
    //     std::cout << itr << std::endl;
    // }
    // std::cout << std::endl;
    //
    // for(const OrganeReparable& itr : OrganeReparable())
    // {
    //     std::cout << (Organe)itr << std::endl;
    // }
    // std::cout << std::endl;
    //
    // for(const auto& itr : OrganeNonReparable())
    // {
    //     std::cout << (Organe)itr << std::endl;
    // }


    // for(const auto& itr : centrale.etats_ouvriers())
    //     std::cout << itr.first << " : " << itr.second << std::endl;
    // for(const auto& itr : centrale.postes_ouvriers())
    //     std::cout << itr.first << " : " << itr.second << std::endl;
    // std::cout << std::endl;
    //
    // centrale.affecter_ouvriers(Organe::Circuit_primaire, 32);
    //
    // for(const auto& itr : centrale.etats_ouvriers())
    //     std::cout << itr.first << " : " << itr.second << std::endl;
    // for(const auto& itr : centrale.postes_ouvriers())
    //     std::cout << itr.first << " : " << itr.second << std::endl;
    // std::cout << std::endl;
    //
    // centrale.desaffecter_ouvriers(Organe::Circuit_primaire);
    //
    // for(const auto& itr : centrale.etats_ouvriers())
    //     std::cout << itr.first << " : " << itr.second << std::endl;
    // for(const auto& itr : centrale.postes_ouvriers())
    //     std::cout << itr.first << " : " << itr.second << std::endl;
    // std::cout << std::endl;


    return EXIT_SUCCESS;
}



// enum class COLORS
// {
//     Blue,
//     Red,
//     Green,
//     Purple,
//     First=Blue,
//     Last=Purple
// } COLOR;
//
// COLORS operator++(COLORS& x) {
//     return x = (COLORS)(std::underlying_type<COLORS>::type(x) + 1);
// }
//
// COLORS operator*(COLORS c) {
//     return c;
// }
//
// COLORS begin(COLORS r) {
//     return COLORS::First;
// }
//
// COLORS end(COLORS r) {
//     COLORS l=COLORS::Last;
//     return ++l;
// }
//
// int main() {
//     //note the parenthesis after COLOR to make an instance
//     for(const auto& c : COLOR) {
//         static int test = 0;
//         std::cout << ++test << std::endl;
//     }
//     return 0;
// }


/*
Update états ouvriers
Affecter postes ouvriers
Update réparations organes
Update centrale
Afficher alertes
*/


// class Organe
// {
// public:
//     enum Reparable
//     {
//         None,
//         Injecteur_acide_borique,
//         Circuit_primaire,
//         Pressuriseur,
//         Pompe_circuit_primaire,
//         Generateur_de_vapeur,
//         Pompe_circuit_secondaire,
//         Circuit_secondaire,
//         Condenseur
//     };
//
//     enum NonReparable
//     {
//         Barre_de_graphite,
//         Canaux,
//         Piscine,
//         Cuve,
//         Echangeur_de_chaleur,
//         Enceinte_de_confinement
//     };
//
//     Organe(Reparable r) : rReparable{r} {}
//     Organe(NonReparable nr) : nrNonReparable{nr} {}
//
// private:
//     Reparable rReparable;
//     NonReparable nrNonReparable;
// };
//
// int main()
// {
//     Organe::NonReparable cuve(Organe::Cuve);
//
//     if(cuve == Organe::Cuve)
//         std::cout << "YOUPI !!! " << std::endl;
//
//
//     return 0;
// }


// class Organe
// {
// public:
//     enum Reparable
//     {
//         None,
//         Injecteur_acide_borique,
//         Circuit_primaire,
//         Pressuriseur,
//         Pompe_circuit_primaire,
//         Generateur_de_vapeur,
//         Pompe_circuit_secondaire,
//         Circuit_secondaire,
//         Condenseur
//     };
//
//     enum NonReparable
//     {
//         Barre_de_graphite,
//         Canaux,
//         Piscine,
//         Cuve,
//         Echangeur_de_chaleur,
//         Enceinte_de_confinement
//     };
//
//     Organe(Reparable r) : rReparable{r} {}
//     Organe(NonReparable nr) : nrNonReparable{nr} {}
//
//     // friend bool operator==(const Organe& lhs, const Organe::NonReparable& rhs)
//     // {
//     //     return lhs == static_cast<Organe>(rhs);
//     // }
//
// private:
//     Reparable rReparable;
//     NonReparable nrNonReparable;
// };


// struct Organe
// {
//     enum Reparable
//     {
//         None,
//         Injecteur_acide_borique,
//         Circuit_primaire,
//         Pressuriseur,
//         Pompe_circuit_primaire,
//         Generateur_de_vapeur,
//         Pompe_circuit_secondaire,
//         Circuit_secondaire,
//         Condenseur
//     };
//
//     enum NonReparable
//     {
//         Barre_de_graphite,
//         Canaux,
//         Piscine,
//         Cuve,
//         Echangeur_de_chaleur,
//         Enceinte_de_confinement
//     };
//
//     friend const bool operator==(Organe::Reparable lhs, Organe::NonReparable rhs) {return false;}
// };
//
// /*
// Organe::Condenseur;
// Organe::Reparable::Condenseur;
// Organe::Cuve;
// Organe::NonReparable::Cuve;
// */
//
// int main()
// {
//     Organe::Reparable condenseur1 = Organe::Condenseur;             // OK
//     Organe::Reparable cuve1 = Organe::Cuve;                         // Erreur
//
//     Organe::NonReparable condenseur2 = Organe::Condenseur;          // Erreur
//     Organe::NonReparable cuve2 = Organe::Cuve;                      // OK
//
//     Organe condenseur3 = Organe::Condenseur;
//     Organe cuve3 = Organe::Cuve;
//
//     Organe condenseur4 = Organe::Reparable::Condenseur;
//     Organe cuve4 = Organe::NonReparable::Cuve;
//
//     if(condenseur1 == Organe::Condenseur)
//         std::cout << "Condenseur 1 : Réparable" << std::endl;
//     if(cuve1 == Organe::Cuve)
//         std::cout << "Cuve 1 : Réparable" << std::endl;
//
//     if(condenseur2 == Organe::Condenseur)
//         std::cout << "Condenseur 2 : Non réparable" << std::endl;
//     if(cuve2 == Organe::Cuve)
//         std::cout << "Cuve  2 : Non réparable" << std::endl;
//
//     if(condenseur3 == Organe::Condenseur)
//         std::cout << "Condenseur 3 : ?" << std::endl;
//     if(cuve3 == Organe::Cuve)
//         std::cout << "Cuve  3 : ?" << std::endl;
//
//     if(condenseur4 == Organe::Condenseur)
//         std::cout << "Condenseur 3 : ?" << std::endl;
//     if(cuve4 == Organe::Cuve)
//         std::cout << "Cuve  3 : ?" << std::endl;
//
//
//
//     return 0;
// }

// struct Organe
// {
//     enum Reparable : short
//     {
//         None,
//         Injecteur_acide_borique,
//         Circuit_primaire,
//         Pressuriseur,
//         Pompe_circuit_primaire,
//         Generateur_de_vapeur,
//         Pompe_circuit_secondaire,
//         Circuit_secondaire,
//         Condenseur
//     };
//
//     enum nonReparable : short
//     {
//         Barre_de_graphite,
//         Canaux,
//         Piscine,
//         Maillot_de_bain,
//         Serviette,
//         Cuve,
//         Echangeur_de_chaleur,
//         Enceinte_de_confinement
//     };
//
//     const Organe::Reparable _r;
//     const Organe::nonReparable _nr;
//
//     Organe() = delete;
//
//     Organe(Organe::Reparable organe) : _r(organe._r) {}
//     Organe(Organe::nonReparable organe) : _nr(organe._nr) {}
// };
//
// /*
// Organe::Condenseur;
// Organe::Reparable::Condenseur;
// Organe::Cuve;
// */
//
// int main()
// {
//     Organe::Reparable condenseur1 = Organe::Condenseur;             // OK
//
//     Organe cuve2 = Organe::Cuve;                                    // OK
//
//     Organe condenseur3 = Organe::Condenseur;
//     Organe cuve3 = Organe::Cuve;
//
//     Organe condenseur4 = Organe::Reparable::Condenseur;
//
//     if(condenseur1 == Organe::Condenseur)
//         std::cout << "Condenseur 1 : Réparable" << std::endl;
//
//     if(cuve2 == Organe::Cuve)
//         std::cout << "Cuve  2 : Non réparable" << std::endl;
//
//     if(condenseur3 == Organe::Condenseur)
//         std::cout << "Condenseur 3 : ?" << std::endl;
//     if(cuve3 == Organe::Cuve)
//         std::cout << "Cuve  3 : ?" << std::endl;
//
//     if(condenseur4 == Organe::Condenseur)
//         std::cout << "Condenseur 3 : ?" << std::endl;
//
//
//
//     return 0;
// }



/*
enum Reparable : short
{
    None,
    Injecteur_acide_borique,
    Circuit_primaire,
    Pressuriseur,
    Pompe_circuit_primaire,
    Generateur_de_vapeur,
    Pompe_circuit_secondaire,
    Circuit_secondaire,
    Condenseur
};

enum nonReparable : short
{
    Barre_de_graphite,
    Canaux,
    Piscine,
    Maillot_de_bain,
    Serviette,
    Cuve,
    Echangeur_de_chaleur,
    Enceinte_de_confinement
};
*/

// struct Organe
// {
//     struct Reparable : Organe
//     {
//
//     };
// };
//
// /*
// Organe::Condenseur;
// Organe::Reparable::Condenseur;
// Organe::Cuve;
// */
//
// int main()
// {
//     Organe organe1;
//     Organe::Reparable organeReparable1;
//
//
//     return 0;
// }
