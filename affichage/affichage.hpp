#pragma once

#include "SDL2/sdl2.hpp"
#include <map>
#include <iostream>
#include "../structures_Centrale/centrale.hpp"




template <class T, long long unsigned N>
std::array<T, N> operator+(std::array<T, N> first, std::array<T, N> second)
{
    std::array<T, N> out;
    for(unsigned i = 0; i < N; ++i)
        out[i] = first[i] + second[i];
    return out;
}

template <class T, long long unsigned N>
std::array<T, N> operator*(std::array<T, N> first, std::array<T, N> second)
{
    std::array<T, N> out;
    for (unsigned i = 0; i < N; ++i)
        out[i] = first[i]*second[i];
    return out;
}

template <class T, long long unsigned N>
std::array<T, N> operator-(std::array<T, N> first, std::array<T, N> second)
{
    std::array<T, N> out;
    for(unsigned i = 0; i < N; ++i)
        out[i] = first[i] - second[i];
    return out;
}

template <class T, long long unsigned N, typename U>
std::array<T, N> operator*(U first, std::array<T, N> second)
{
    std::array<T, N> out;
    for(unsigned i = 0; i < N; ++i)
        out[i] = first*second[i];
    return out;
}

namespace affichage
{
    enum class modes : unsigned char {Void, Mode1, Mode2, Mode3};

    template <class T>
    using point = std::array<T, 2ULL>;
    using rectangle = point<point<int>>;

    typedef std::pair<std::string, rectangle> paire;
    class CurrentRenders
    {
        public:

            CurrentRenders();
            
            template <class Paire, class ... Paires> CurrentRenders(const Paire &, const Paires &...);
            CurrentRenders(const paire &);  // Force le type du constructeur variadique
            void deleteByFilter(const std::string &);
            void ajouterRender(const paire &);
            rectangle getElementById(const std::string &) noexcept;

            void afficher() noexcept;
            void render(sdl2::window &);


            /**
             * 
             * @brief Retourne la clef de l'element sur lequel le curseur pt est, on suppose ici qu'il n'y a aucun enchevetrement
             * @param
             *        std::array<int, 2ULL> pt : point teste
             * @retval bool si pt est dans rect
             */
            const std::string where(const point<int> &);

            /**
             * 
             * @brief Verifie si pt est a l'interieur de l'unique rectangle defini par les points extremes 
             *        (donnes dans l'ordre croissant lexicographique).
             * @param
             *        std::array<int, 2ULL> pt : point teste
             *        rectangle rect : rectangle (decrit par ses deux points extremes)
             * @retval bool si pt est dans rect
             */
            friend inline bool isOn(const point<int> &, const rectangle &) noexcept;

        private:
            std::map<std::string, rectangle> Elements;          /* Dictionnaire qui permet de stocker les positions de chaque objet actuellement
                                                                                                affiche a l'ecran.
                                                                                             */

    };

    class Barre
    {
        public:
            Barre(const std::string & name, double vmin, double vmax, double vact, point<int> at,
                        point<int> dimensions, const std::string & police = "./data/Lato-Medium.ttf");
            void render(sdl2::window & fenetre);
            inline double pourcentageActuel() noexcept;
            inline rectangle getZone() noexcept;
            inline double getValeurMin() noexcept;
            inline double getValeurMax() noexcept;
            inline double getValeurActuelle() noexcept;
            inline std::string getNom() noexcept;
            inline std::string getFont() noexcept;
            inline double setValeurActuelle(double) noexcept;
        private:
            std::string nom;
            std::string font;
            double valeur_min;
            double valeur_max;
            double valeur_actuelle;
            rectangle zone;
    };

    class BarreEtat : public Barre
    {
        public:
            BarreEtat(const std::string & name, double vmin, double vmax, double vact, point<int> at,
                        point<int> dimensions, const std::string & police = "./data/Lato-Medium.ttf") : Barre(name, vmin, vmax, vact, at, dimensions, police){}
            virtual void render(sdl2::window & fenetre);
    };
    class Timer : public BarreEtat
    {
        public:
            Timer(point<int> at, point<int> dimensions) : BarreEtat("TIMER", 0, 1, 0, at, dimensions, "./data/Lato-Medium.ttf"), nombre_de_tours(0){}
            void render(sdl2::window & fenetre) final;
            unsigned & setNombre_de_tours(const unsigned nbt);
        private:
            unsigned nombre_de_tours;
    };

    class BarreModifiable : public Barre
    {
        public:
            BarreModifiable(const std::string & name, double vmin, double vmax, double vact, double vdem, 
                            double va_min, double va_max, point<int> at, point<int> dimensions, const std::string & police = "./data/Lato-Medium.ttf");

            inline double setValeurDemandee(double vdem);
            inline double setValeurDemandee(point<int> position);
            inline double setValeurActuelleMin(double vact_min);
            inline double setValeurActuelleMax(double vact_min);
            inline double getValeurDemandee() noexcept;
            inline double pourcentageDemande() noexcept;
            inline double pourcentageActuelMin() noexcept;
            inline double pourcentageActuelMax() noexcept;
            void render(sdl2::window & fenetre);
        private:
            volatile double valeur_demandee;
            double valeur_actuelle_min;
            double valeur_actuelle_max;
    };

    class PanneauAlertes
    {
        public:
            PanneauAlertes(map_alertes_t Alertes, const std::string police, rectangle z) : alertes(Alertes), font(police), zone(z) {}
            const map_alertes_t & setAlertes(map_alertes_t &);
            void render(sdl2::window & fenetre);
        private:
            map_alertes_t alertes;
            std::string font;
            rectangle zone;
    };

    class Button
    {
        public:
            enum class etat : unsigned {Inactive, Active, Enfonce};

            std::string PATH_INACTIVE;
            std::string PATH_ACTIVE;
            std::string PATH_ENFONCE;      

            Button(point<int> at, point<int> dimensions, etat e = etat::Inactive) : Etat(e), zone(rectangle({at, at + dimensions})) {};
            Button(const Button & B);
            inline etat setEtat(etat e);
            inline etat getEtat();
            const std::string & getCurPath();
            void render(sdl2::window & fenetre);
            rectangle getZone();
            point<int> getPosition();
            point<int> dimensions();
        
        private:
            etat Etat;
            rectangle zone;
    };

    class ButtonStop : public Button
    {
        public:
            
            ButtonStop(point<int> at, point<int> dimensions, etat e = etat::Inactive) : Button(at, dimensions, e)
            {
                PATH_INACTIVE = "data/shutdown_inactif.png";
                PATH_ACTIVE = "data/shutdown_actif.png";
                PATH_ENFONCE = "data/shutdown_press.png";
            }

    };

    class ButtonPopulation : public Button
    {
        public:
        
            ButtonPopulation(point<int> at, point<int> dimensions, etat e = etat::Inactive) : Button(at, dimensions, e) 
            {
                PATH_INACTIVE = "data/evacpop_inactif.png";
                PATH_ACTIVE = "data/Jeanne.png";
                PATH_ENFONCE = "data/evacpop_press.png";
            };
    };

    class IllustrationCentrale
    {
        public:
            std::string schema = "data/Illustration/schema.png";

            std::string pressuriseur = "data/Illustration/pressuriseur.png";
            std::string pompe_circuit_primaire = "data/Illustration/pompe_circuit_primaire.png";
            std::string generateur_de_vapeur = "data/Illustration/générateur_de_vapeur.png";
            std::string circuit_primaire = "data/Illustration/circuit_primaire.png";
            std::string circuit_secondaire = "data/Illustration/circuit_secondaire.png";
            std::string cuve = "data/Illustration/cuve.png";
            std::string barres_de_graphite = "data/Illustration/barres_de_graphite.png";
            std::string pompe_circuit_secondaire = "data/Illustration/pompe_circuit_secondaire.png";
            std::string condenseur = "data/Illustration/condenseur.png";
            std::string enceinte_de_confinement = "data/Illustration/enceinte_de_confinement.png";
            std::string echangeur_de_chaleur = "data/Illustration/echangeur_de_chaleur.png";


            IllustrationCentrale(std::array<bool, 11ULL> touches, point<int> position, point<int> dimensions, unsigned per = 5, unsigned compt = 0);
            std::array<bool, 11ULL> & set(std::array<bool, 11ULL> touches);
            point<int> getPosition();
            point<int> dimension();
            rectangle getZone();
            unsigned getPeriode(){return periode;}
            unsigned getCompteur(){return compteur;}

            void update(){compteur = ((compteur < periode - 1) ? compteur + 1 : 0);}
            void render(sdl2::window & fenetre);

        private:
            std::array<bool, 11ULL> organes_touches;    // Dans cet ordre : pressuriseur pompe_circuit_primaire generateur_de_vapeur circuit_primaire circuit_secondaire piscine cuve barres_de_graphite pompe_circuit_secondaire condenseur
            rectangle zone;
            unsigned periode;
            unsigned compteur;
    };

    class ButtonsOuvriers
    {
      
        public:
            class ButtonMore : public Button
            {
                public:
                    ButtonMore(point<int> at, point<int> dimensions, etat e = etat::Inactive) : Button(at, dimensions, e)
                    {
                        PATH_INACTIVE = "data/Illustration/plus_inactif.png";
                        PATH_ACTIVE = "data/Illustration/plus_enfonce.png";
                        PATH_ENFONCE = "data/Illustration/plus_enfonce.png";
                    }
            };
            class ButtonLess : public Button
            {
                public:
                    ButtonLess(point<int> at, point<int> dimensions, etat e = etat::Inactive) : Button(at, dimensions, e)
                    {
                        PATH_INACTIVE = "data/Illustration/moins_inactif.png";
                        PATH_ACTIVE = "data/Illustration/moins_enfonce.png";
                        PATH_ENFONCE = "data/Illustration/moins_enfonce.png";
                    }
            };
            class ButtonGo : public Button
            {
                public:
                    ButtonGo(point<int> at, point<int> dimensions, etat e = etat::Inactive) : Button(at, dimensions, e)
                    {
                        PATH_INACTIVE = "data/Illustration/go_inactif.png";
                        PATH_ACTIVE = "data/Illustration/go_actif.png";     // Permet de savoir si l'envoi des ouvriers est possible
                        PATH_ENFONCE = "data/Illustration/go_enfonce.png";
                    }
            };
            class ButtonUndo : public Button
            {
                public:
                    ButtonUndo(point<int> at, point<int> dimensions, etat e = etat::Inactive) : Button(at, dimensions, e)
                    {
                        PATH_INACTIVE = "data/Illustration/undo_inactif.png";
                        PATH_ACTIVE = "data/Illustration/undo_enfonce.png";
                        PATH_ENFONCE = "data/Illustration/undo_enfonce.png";
                    }
            };


            ButtonsOuvriers(const point<int> & at, const point<int> & dimensions, unsigned nb_o_d);
            void render(sdl2::window & fenetre);
            const unsigned getNb_ouvriers(){return nb_ouvriers;}
            unsigned & setNb_ouvriers(unsigned nb);
            unsigned & setNb_ouvriers_envoyes(unsigned);
            void setEtats(std::array<Button::etat, 4ULL>);
            ButtonsOuvriers & operator++();
            ButtonsOuvriers & operator--();

            ButtonMore & getBouton_plus();
            ButtonLess & getBouton_moins();
            ButtonGo & getBouton_envoyer();
            ButtonUndo & getBouton_renvoyer();

            ButtonsOuvriers & rapatrier();
            ButtonsOuvriers & envoyer();
            
            static unsigned nb_ouvriers_disponibles;

        private:
            ButtonMore bouton_plus;
            ButtonLess bouton_moins;
            ButtonGo bouton_envoyer;
            ButtonUndo bouton_renvoyer;

            rectangle zone;

            const unsigned nb_ouvriers_defaut;
            unsigned nb_ouvriers;
            unsigned nb_ouvriers_envoyes;

            
    };
}

// Classe permettant de stocker les informations utiles pour l'affichage de l'écran de fin de jeu
class Jeu
{
    private:
        unsigned nombre_de_tours;
        double production_cum;
        affichage::modes Mode;
        std::size_t score;

    public:
        Jeu() : nombre_de_tours(0), production_cum(0), Mode(affichage::modes::Void), score(100), fin_du_game(false) {}
        
        void update(double production_actuelle, std::size_t score_actuel = 100);

        unsigned getNombre_de_tours();
        
        double getProduction_moyenne();

        std::size_t getScore();
        
        affichage::modes & setMode(affichage::modes M);
        affichage::modes getMode();
        bool fin_du_game;
};

namespace plagedonnees
{
    static constexpr const double _min_T_press = 25; 
    static constexpr const double _min_F_p1 = 0.;
    static constexpr const double _max_Tx_Gr = 1.; 
    static constexpr const double _min_Tx_B = 0.; 
    static constexpr const double _max_Tx_B = 0.5;
    static constexpr const double _min_E_p2 = 0.;
    static constexpr const double _min_F_cd = 0.;


    const double min_T_press(Centrale & centrale) {return _min_T_press;}
    const double max_T_press(Centrale & centrale) {return 100*(centrale.circuit_primaire.E_press() + centrale.circuit_primaire.E_res());}

    const double min_F_p1(Centrale & centrale) {return _min_F_p1;}
    const double max_F_p1(Centrale & centrale) {return centrale.circuit_primaire.E_p1();}

    const double min_Tx_Gr(Centrale & centrale) {return (1-centrale.reacteur.E_barre());}
    const double max_Tx_Gr(Centrale & centrale) {return _max_Tx_Gr;}

    const double min_Tx_B(Centrale & centrale) {return _min_Tx_B;}
    const double max_Tx_B(Centrale & centrale) {return _max_Tx_B;}

    const double min_E_p2(Centrale & centrale) {return _min_E_p2;}
    const double max_E_p2(Centrale & centrale) {return centrale.circuit_secondaire.E_p2();}

    const double min_F_cd(Centrale & centrale) {return _min_F_cd;}
    const double max_F_cd(Centrale & centrale) {return centrale.circuit_secondaire.E_cd();}
}
