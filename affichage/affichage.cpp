#include "SDL2/sdl2.hpp"
#include "affichage.hpp"
#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include "../structures_Centrale/centrale.hpp"
#include "../structures_Centrale/dispatching.hpp"


// Définition de la constante statique donnant la limite d'ouvriers disponibles
unsigned affichage::ButtonsOuvriers::nb_ouvriers_disponibles = 145;

std::ostream & operator<<(std::ostream & os, sdl2::color c)
{
    os << "Rouge : " << c.red << " Vert : " << c.green << " Bleu : " << c.blue << " Alpha : " << c.alpha;
    return os;
}

template <class T, long long unsigned N>
std::ostream & operator<<(std::ostream & os, std::array<T, N> M)
{
    os << "[";
    for(unsigned i = 0; i < N - 1; ++i)
        os << M[i] << " ";
    os << M[N - 1] << "]";
    return os;
}


template <class Paire, class ... Paires>
affichage::CurrentRenders::CurrentRenders(const Paire & image_id1, const Paires &... images_ids) : CurrentRenders(images_ids...)
{
    affichage::point<int> vec1{image_id1.second[0][0], image_id1.second[0][1]};
    affichage::point<int> vec2{image_id1.second[0][0] + image_id1.second[1][0], image_id1.second[0][1] + image_id1.second[1][1]};
    rectangle Rect = {vec1, vec2};
    Elements.insert(std::pair<std::string, rectangle>(image_id1.first, Rect));
}

affichage::CurrentRenders::CurrentRenders(const affichage::paire & image_id)      // Forçage du type d'entree à paire
{
    affichage::point<int> vec1({image_id.second[0][0], image_id.second[0][1]});
    affichage::point<int> vec2{image_id.second[0][0] + image_id.second[1][0], image_id.second[0][1] + image_id.second[1][1]};
    rectangle Rect = {vec1, vec2};
    Elements.insert(std::pair<std::string, rectangle>(image_id.first, Rect));
}

void affichage::CurrentRenders::ajouterRender(const affichage::paire & image_id)
{
    affichage::point<int> vec1({image_id.second[0][0], image_id.second[0][1]});
    affichage::point<int> vec2{image_id.second[0][0] + image_id.second[1][0], image_id.second[0][1] + image_id.second[1][1]};
    rectangle Rect = {vec1, vec2};
    Elements.insert(std::pair<std::string, rectangle>(image_id.first, Rect));
}

void affichage::CurrentRenders::afficher() noexcept
{
    for (auto [first, second] : Elements)
    {
        std::cout << first << " :\nCote bas en " << second[0][0] << ", " << second[0][1] << std::endl;
        std::cout << "Cote haut en " << second[1][0] << ", " << second[1][1] << std::endl;
    }
}

void affichage::CurrentRenders::render(sdl2::window & fenetre)
{
    for (auto [first, second] : Elements)
    {
        sdl2::rectangle rect(second[0], second[1] - second[0], {255, 0, 0, 0}, false);
        fenetre << rect << sdl2::flush;
    }
}

affichage::rectangle affichage::CurrentRenders::getElementById(const std::string & Id) noexcept
{
    return Elements[Id];
}


inline bool isOn(const affichage::point<int> & pt, const affichage::rectangle & rect) noexcept
{
    return (rect[0][0] <= pt[0]) && (pt[0] <= rect[1][0]) && (rect[0][1] <= pt[1]) && (pt[1] <= rect[1][1]);
}


const std::string affichage::CurrentRenders::where(const affichage::point<int> & pos)
{
    for (auto [first, second] : Elements)
    {
        if (isOn(pos, second))
            return first;
    }
    return "";
}

void affichage::CurrentRenders::deleteByFilter(const std::string & subIdentifier)
{
    for (std::map<std::string, rectangle>::iterator it = Elements.begin(); it!=Elements.end();)
    {

        if (it->first.find(subIdentifier) != std::string::npos)
        {
            Elements.erase((it++)->first);
        }
        else
        {
            it++;
        }
    }
}

affichage::Barre::Barre(const std::string & name, double vmin, double vmax, double vact, affichage::point<int> at, affichage::point<int> dimensions, const std::string & police)
{
    if((vmin >= vmax) || (vact > vmax) || (vact < vmin)) throw std::invalid_argument("Bornes incorrectes ou données invalides");
    nom = name;
    font = police;
    valeur_actuelle = vact;
    valeur_min = vmin;
    valeur_max = vmax;
    std::array<int, 2ULL> p2;
    p2 = {at[0] + dimensions[0], at[1] + dimensions[1]};
    zone = {at, p2};
}

inline double affichage::Barre::setValeurActuelle(double vact) noexcept
{
    if (vact < valeur_min)
        return valeur_actuelle = valeur_min;
    if (vact > valeur_max)
        return valeur_actuelle = valeur_max;
    return valeur_actuelle = vact;
}

inline double affichage::Barre::getValeurActuelle() noexcept
{
    return valeur_actuelle;
}

inline double affichage::Barre::getValeurMin() noexcept
{
    return valeur_min;
}

inline double affichage::Barre::getValeurMax() noexcept
{
    return valeur_max;
}

inline std::string affichage::Barre::getNom() noexcept
{
    return nom;
}

inline std::string affichage::Barre::getFont() noexcept
{
    return font;
}

inline affichage::rectangle affichage::Barre::getZone() noexcept
{
    return zone;
}

inline double affichage::Barre::pourcentageActuel() noexcept
{
    return (valeur_actuelle - valeur_min)/(valeur_max - valeur_min);
}

void affichage::Barre::render(sdl2::window & fenetre)
{
    affichage::point<int> dimension_ext = zone[1] - zone[0];
    affichage::point<int> dimension_int = {static_cast<int>(dimension_ext[0]*pourcentageActuel()), dimension_ext[1] - 2};

    sdl2::color blanc = {255, 255, 255, 0};
    sdl2::color gris = {80, 80, 80, 0};

    affichage::point<int> pos_int = {zone[0][0] + 1, zone[0][1] + 1};
    affichage::point<int> pos_ext = {zone[0][0], zone[0][1]};

    sdl2::rectangle rect_cache(pos_ext, dimension_ext, {0, 0, 0, 0}, true);
    sdl2::rectangle rect_ext(pos_ext, dimension_ext, blanc, false);
    sdl2::rectangle rect_int(pos_int , dimension_int, gris, true);

    sdl2::font police(font, dimension_ext[1]);
    sdl2::font policeLabel(font, dimension_ext[1]*0.8);
    sdl2::texte Presentation(nom, policeLabel, fenetre, blanc);
    sdl2::texte Tvmin(std::to_string(static_cast<int>(valeur_min)), police, fenetre, blanc);
    sdl2::texte Tvmax(std::to_string(static_cast<int>(valeur_max)), police, fenetre, blanc);
    Presentation.at(zone[0][0] + dimension_ext[0]/2 - Presentation.dimension()[0]/2, zone[0][1] - 1.25*dimension_ext[1]);

    Tvmin.at(pos_ext[0] - 1.3*Tvmin.dimension()[0], pos_ext[1] + (dimension_ext[1] - Tvmin.dimension()[1])/2);
    Tvmax.at(zone[1][0] + Tvmin.dimension()[0]/2, pos_ext[1] + (dimension_ext[1] - Tvmax.dimension()[1])/2);

    fenetre << rect_cache << rect_ext << rect_int << Presentation << Tvmin << Tvmax;
}

void affichage::BarreEtat::render(sdl2::window & fenetre)
{
    auto zone = getZone();
    double pourcentage = pourcentageActuel();
    affichage::point<int> dimension_ext = zone[1] - zone[0];
    affichage::point<int> dimension_int = {static_cast<int>((dimension_ext[0])*pourcentage), dimension_ext[1] - 2};

    sdl2::color blanc = {255, 255, 255, 0};
    sdl2::color rouge = {255, 0, 0, 0};
    sdl2::color vert = {0, 255, 0, 0};

    sdl2::color couleur_etat = vert*pourcentage + rouge*(1 - pourcentage);

    affichage::point<int> pos_int = {zone[0][0] + 1, zone[0][1] + 1};
    affichage::point<int> pos_ext = {zone[0][0], zone[0][1]};

    sdl2::rectangle rect_cache(pos_ext, dimension_ext, {0, 0, 0, 0}, true);
    sdl2::rectangle rect_ext(pos_ext, dimension_ext, blanc, false);
    sdl2::rectangle rect_int(pos_int, dimension_int, couleur_etat, true);

    sdl2::font police(getFont(), dimension_ext[1]);
    sdl2::font policeLabel(getFont(), dimension_ext[1]*0.8);
    sdl2::texte Presentation(getNom(), policeLabel, fenetre, blanc);
    sdl2::texte Tvmin(std::to_string(static_cast<int>(getValeurMin())), police, fenetre, blanc);
    sdl2::texte Tvmax(std::to_string(static_cast<int>(getValeurMax())), police, fenetre, blanc);
    Presentation.at(zone[0][0] + dimension_ext[0]/2 - Presentation.dimension()[0]/2, zone[0][1] - dimension_ext[1]*1.25);

    Tvmin.at(pos_ext[0] - 1.3*Tvmin.dimension()[0], pos_ext[1] + (dimension_ext[1] - Tvmin.dimension()[1])/2);
    Tvmax.at(zone[1][0] + Tvmin.dimension()[0]/2, pos_ext[1] + (dimension_ext[1] - Tvmax.dimension()[1])/2);

    fenetre << rect_cache << rect_ext << rect_int << Presentation << Tvmin << Tvmax;
}

unsigned & affichage::Timer::setNombre_de_tours(const unsigned nbt)
{
    return nombre_de_tours = nbt;
}

void affichage::Timer::render(sdl2::window & fenetre)
{
    auto zone = getZone();
    double pourcentage = pourcentageActuel();
    affichage::point<int> dimension_ext = zone[1] - zone[0];
    affichage::point<int> dimension_int = {static_cast<int>((dimension_ext[0])*pourcentage), dimension_ext[1] - 2};

    sdl2::font police(getFont(), 3*dimension_ext[1]);

    sdl2::color blanc = {255, 255, 255, 0};
    sdl2::color rouge = {255, 0, 0, 0};
    sdl2::color vert = {0, 255, 0, 0};
    sdl2::color noir = {0, 0, 0, 255};

    sdl2::texte tours("Tour : " + std::to_string(nombre_de_tours), police, fenetre, blanc);

    tours.at(zone[1][0] + dimension_ext[0]/5, zone[0][1] - tours.dimension()[1]/2);

    sdl2::color couleur_etat = vert*pourcentage + rouge*(1 - pourcentage);

    affichage::point<int> pos_int = {zone[0][0] + 1, zone[0][1] + 1};
    affichage::point<int> pos_ext = {zone[0][0], zone[0][1]};

    sdl2::rectangle rect_cache(pos_ext, dimension_ext, {0, 0, 0, 0}, true);
    sdl2::rectangle rect_ext(pos_ext, dimension_ext, blanc, false);
    sdl2::rectangle rect_int(pos_int, dimension_int, couleur_etat, true);
    sdl2::rectangle cache_tours(tours.getPosition(), tours.dimension(), noir, true);

    fenetre << rect_cache << rect_ext << rect_int << cache_tours << tours;
}

affichage::BarreModifiable::BarreModifiable(const std::string & name, double vmin, double vmax, double vact, double vdem, double va_min, double va_max, affichage::point<int> at, affichage::point<int> dimensions, const std::string & police) : Barre(name, vmin, vmax, vact, at, dimensions, police)
{
    if ((vdem > vmax) || (vdem < vmin) || (va_min >= va_max))
        throw std::invalid_argument("Donnée demandée invalide ou valeurs limites invalides.");

    valeur_demandee = vdem;
    valeur_actuelle_max = va_max;
    valeur_actuelle_min = va_min;
}

inline double affichage::BarreModifiable::pourcentageDemande() noexcept
{
    return (valeur_demandee - affichage::Barre::getValeurMin())/(affichage::Barre::getValeurMax() - affichage::Barre::getValeurMin());
}

inline double affichage::BarreModifiable::setValeurDemandee(double vdem)
{
    if (vdem > valeur_actuelle_max)
        return valeur_demandee = valeur_actuelle_max;
    if (vdem < valeur_actuelle_min)
        return valeur_demandee = valeur_actuelle_min;
    return valeur_demandee = vdem;
}

inline double affichage::BarreModifiable::setValeurActuelleMin(double vact_min)
{
    if (vact_min > valeur_actuelle_max)
        return valeur_actuelle_min = valeur_actuelle_max;
    if (vact_min < getValeurMin())
        return valeur_actuelle_min = getValeurMin();
    return valeur_actuelle_min = vact_min;
}

inline double affichage::BarreModifiable::setValeurActuelleMax(double vact_max)
{
    if (vact_max > valeur_actuelle_max)
        return valeur_actuelle_max = getValeurMax();
    if (vact_max < valeur_actuelle_min)
        return valeur_actuelle_max = valeur_actuelle_min;
    return valeur_actuelle_max = vact_max;
}


inline double affichage::BarreModifiable::setValeurDemandee(point<int> position)
{
    auto zone = getZone();
    double inf = getValeurMin();
    if ((position[1] >= zone[0][1]) && (position[1] <= zone[1][1]))
        setValeurDemandee(static_cast<double>((position[0] - zone[0][0]))/(zone[1][0] - zone[0][0])*(getValeurMax() - inf) + inf);
    return valeur_demandee;
}

inline double affichage::BarreModifiable::getValeurDemandee() noexcept
{
    return valeur_demandee;
}

inline double affichage::BarreModifiable::pourcentageActuelMin() noexcept
{
    return (valeur_actuelle_min - getValeurMin())/(getValeurMax() - getValeurMin());
}

inline double affichage::BarreModifiable::pourcentageActuelMax() noexcept
{
    return (valeur_actuelle_max - getValeurMin())/(getValeurMax() - getValeurMin());
}

void affichage::BarreModifiable::render(sdl2::window & fenetre)
{
    double valeur_min, valeur_max, pourcentage_demande, pourcentage_minimal, pourcentage_maximal, pourcentage_actuel;
    valeur_min = getValeurMin();
    valeur_max = getValeurMax();
    pourcentage_demande = pourcentageDemande();
    pourcentage_minimal = pourcentageActuelMin();
    pourcentage_maximal = pourcentageActuelMax();
    pourcentage_actuel = pourcentageActuel();

    rectangle zone = getZone();
    const std::string font = getFont();
    const std::string nom = getNom();

    sdl2::color blanc = {255, 255, 255, 0};
    sdl2::color bleu = {0, 0, 255, 0};
    sdl2::color rouge = {255, 0, 0, 0};
    sdl2::color gris = {80, 80, 80, 0};

    affichage::point<int> dimension_ext = zone[1] - zone[0];
    affichage::point<int> dimension_int = dimension_ext - std::array<int, 2>({2, 2});

    affichage::point<int> pos_int = zone[0] + std::array<int, 2>({1, 1});
    affichage::point<int> pos_ext = zone[0];

    affichage::point<int> tmp1 = {static_cast<int>(pourcentage_minimal*dimension_ext[0]), dimension_int[1]};

    affichage::point<int> tmp21 = {static_cast<int>(pourcentage_maximal*(dimension_ext[0] - 1)), 0};
    affichage::point<int> tmp22 = {static_cast<int>((1 - pourcentage_maximal)*(dimension_ext[0] - 1)), dimension_int[1]};

    sdl2::rectangle rect_ext(pos_ext, dimension_ext, blanc, false);
    sdl2::rectangle rect_int(pos_int, dimension_int, bleu, true);
    sdl2::rectangle marge_grisee1(pos_int, tmp1, gris, true);
    sdl2::rectangle marge_grisee2(pos_int + tmp21, tmp22, gris, true);

    double res_Vmin = valeur_min - std::ceil(valeur_min);
    double res_Vmax = valeur_max - std::ceil(valeur_max);

    std::string ch_Vmin = std::to_string(res_Vmin);
    std::string ch_Vmax = std::to_string(res_Vmax);

    std::string Vmin;
    std::string Vmax;

    Vmin = std::abs(res_Vmin) > 0.1 ? std::to_string(static_cast<int>(valeur_min)) + "." + ch_Vmin[3] : Vmin = std::to_string(static_cast<int>(valeur_min));
    Vmax = std::abs(res_Vmax) > 0.1 ? std::to_string(static_cast<int>(valeur_max)) + "." + ch_Vmax[3] : Vmax = std::to_string(static_cast<int>(valeur_max));

    sdl2::font police(font, dimension_ext[1]);
    sdl2::font policeLabel(font, dimension_ext[1]*0.8);
    sdl2::texte Presentation(nom, policeLabel, fenetre, blanc);
    sdl2::texte Tvmin(Vmin, police, fenetre, blanc);
    sdl2::texte Tvmax(Vmax, police, fenetre, blanc);
    Presentation.at(zone[0][0] + dimension_ext[0]/2 - Presentation.dimension()[0]/2, zone[0][1] - dimension_ext[1]*1.25);

    Tvmin.at(pos_ext[0] - 1.3*Tvmin.dimension()[0], pos_ext[1] + (dimension_ext[1] - Tvmin.dimension()[1])/2);
    Tvmax.at(zone[1][0] + Tvmin.dimension()[0]/2, pos_ext[1] + (dimension_ext[1] - Tvmax.dimension()[1])/2);

    affichage::point<int> pos1_depart = {static_cast<int>(zone[0][0] + 1 + pourcentage_demande*dimension_ext[0]), zone[0][1] + 1};
    affichage::point<int> pos1_arrivee = {static_cast<int>(zone[0][0] + 1 + pourcentage_demande*dimension_ext[0]), zone[0][1] + 1 + dimension_int[1]};

    affichage::point<int> pos2_depart = {static_cast<int>(zone[0][0] + 1 + pourcentage_actuel*dimension_ext[0]), zone[0][1] + 1};
    affichage::point<int> pos2_arrivee = {static_cast<int>(zone[0][0] + 1 + pourcentage_actuel*dimension_ext[0]), zone[0][1] + 1 + dimension_int[1]};

    sdl2::line ligne_voulue(pos1_depart, pos1_arrivee, rouge);
    sdl2::line ligne_realisee(pos2_depart, pos2_arrivee, blanc);

    fenetre << rect_ext << rect_int << Presentation << Tvmin << Tvmax << marge_grisee1 << marge_grisee2 << ligne_voulue << ligne_realisee;
}


const map_alertes_t & affichage::PanneauAlertes::setAlertes(map_alertes_t & Alertes)
{
    alertes = Alertes;
    return Alertes;
}

void affichage::PanneauAlertes::render(sdl2::window & fenetre)
{
    point<int> border = {10, 10};
    double facteur = 1.2;
    unsigned compteur = 0;
    affichage::point<int> Position = zone[0];
    affichage::point<int> Dimension = zone[1] - zone[0];
    sdl2::rectangle Background(Position - facteur*border, Dimension + 2*facteur*border, {40, 40, 40, 0}, false);
    sdl2::font Font(font, 0.06*(zone[1] - zone[0])[1]);

    std::string Alertes;
    for (auto & it : alertes)
    {
        if (compteur > (unsigned)(facteur/0.12))
        {
            Alertes += "...";
            break;
        }
        Alertes += toString(it.first) + " : " + it.second + "\n";
        compteur++;
    }
    sdl2::texte Texte(Alertes, Font, fenetre, Dimension[0], {255, 0, 0, 0}, {0, 0, 0, 0});
    Texte.at(zone[0][0], zone[0][1]);
    fenetre << Texte << Background;
}

affichage::Button::Button(const affichage::Button & B)
{
    zone = getZone();
    Etat = getEtat();
}

inline affichage::ButtonStop::etat affichage::Button::setEtat(affichage::Button::etat e)
{
    return Etat = e;
}

inline affichage::rectangle affichage::Button::getZone()
{
    return zone;
}

inline affichage::Button::etat affichage::Button::getEtat()
{
    return Etat;
}

void affichage::Button::render(sdl2::window & fenetre)
{
    sdl2::image img(getCurPath(), fenetre);
    img.stretch((getZone()[1] - getZone()[0]));
    img.at(getZone()[0][0], getZone()[0][1]);
    fenetre << img;
};

const std::string & affichage::Button::getCurPath()
{
    switch (getEtat())
    {
        case etat::Inactive :
            return PATH_INACTIVE;
        case etat::Enfonce :
            return PATH_ENFONCE;
        case etat::Active :
            return PATH_ACTIVE;
    }
    return PATH_INACTIVE;   // Encore pour faire plaisir au compilateur
}

affichage::point<int> affichage::Button::getPosition()
{
    return zone[0];
}

affichage::point<int> affichage::Button::dimensions()
{
    return zone[1] - zone[0];
}

affichage::IllustrationCentrale::IllustrationCentrale(std::array<bool, 11ULL> touches, affichage::point<int> position, affichage::point<int> dimensions, unsigned per, unsigned compt)
{
    zone = {position, position + dimensions};
    organes_touches = touches;
    compteur = compt;
    periode = per;
}

std::array<bool, 11ULL> & affichage::IllustrationCentrale::set(std::array<bool, 11ULL> touches)
{
    organes_touches = touches;
    return organes_touches;
}

void affichage::IllustrationCentrale::render(sdl2::window & fenetre)
{
    double rapport = static_cast<double>(compteur)/periode;
    sdl2::image centrale(schema, fenetre);
    centrale.at(zone[0][0], zone[0][1]);
    centrale.stretch(zone[1] - zone[0]);
    sdl2::color surbrillance = {static_cast<int>(255*rapport), 0, 0, static_cast<int>(255*rapport)};
    fenetre << centrale;

    if (organes_touches[0] == true)
    {
        sdl2::image Pressuriseur(pressuriseur, fenetre, surbrillance);
        Pressuriseur.at(zone[0][0], zone[0][1]);
        Pressuriseur.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Pressuriseur;
    }
    if (organes_touches[1] == true)
    {
        sdl2::image Pompe_circuit_primaire(pompe_circuit_primaire, fenetre, surbrillance);
        Pompe_circuit_primaire.at(zone[0][0], zone[0][1]);
        Pompe_circuit_primaire.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Pompe_circuit_primaire;
    }
    if (organes_touches[2] == true)
    {
        sdl2::image Generateur_de_vapeur(generateur_de_vapeur, fenetre, surbrillance);
        Generateur_de_vapeur.at(zone[0][0], zone[0][1]);
        Generateur_de_vapeur.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Generateur_de_vapeur;
    }
    if (organes_touches[3] == true)
    {
        sdl2::image Circuit_primaire(circuit_primaire, fenetre, surbrillance);
        Circuit_primaire.at(zone[0][0], zone[0][1]);
        Circuit_primaire.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Circuit_primaire;
    }
    if (organes_touches[4] == true)
    {
        sdl2::image Circuit_secondaire(circuit_secondaire, fenetre, surbrillance);
        Circuit_secondaire.at(zone[0][0], zone[0][1]);
        Circuit_secondaire.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Circuit_secondaire;
    }
    if (organes_touches[5] == true)
    {
        sdl2::image Cuve(cuve, fenetre, surbrillance);
        Cuve.at(zone[0][0], zone[0][1]);
        Cuve.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Cuve;
    }
    if (organes_touches[6] == true)
    {
        sdl2::image Barres_de_graphite(barres_de_graphite, fenetre, surbrillance);
        Barres_de_graphite.at(zone[0][0], zone[0][1]);
        Barres_de_graphite.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Barres_de_graphite;
    }
    if (organes_touches[7] == true)
    {
        sdl2::image Pompe_circuit_secondaire(pompe_circuit_secondaire, fenetre, surbrillance);
        Pompe_circuit_secondaire.at(zone[0][0], zone[0][1]);
        Pompe_circuit_secondaire.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Pompe_circuit_secondaire;
    }
    if (organes_touches[8] == true)
    {
        sdl2::image Condenseur(condenseur, fenetre, surbrillance);
        Condenseur.at(zone[0][0], zone[0][1]);
        Condenseur.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Condenseur;
    }
    if (organes_touches[9] == true)
    {
        sdl2::image Enceinte_de_confinement(enceinte_de_confinement, fenetre, surbrillance);
        Enceinte_de_confinement.at(zone[0][0], zone[0][1]);
        Enceinte_de_confinement.stretch(zone[1] - zone[0]).color_filter(surbrillance);;
        fenetre << Enceinte_de_confinement;
    }
    if (organes_touches[10] == true)
    {
        sdl2::image Echangeur_de_chaleur(echangeur_de_chaleur, fenetre, surbrillance);
        Echangeur_de_chaleur.at(zone[0][0], zone[0][1]);
        Echangeur_de_chaleur.stretch(zone[1] - zone[0]).color_filter(surbrillance);
        fenetre << Echangeur_de_chaleur;
    }
    fenetre << sdl2::flush;
}

affichage::point<int> affichage::IllustrationCentrale::getPosition()
{
    return zone[0];
}

affichage::point<int> affichage::IllustrationCentrale::dimension()
{
    return zone[1] - zone[0];
}

affichage::rectangle affichage::IllustrationCentrale::getZone()
{
    return zone;
}

affichage::ButtonsOuvriers::ButtonsOuvriers(const affichage::point<int> & at, const affichage::point<int> & dimensions, unsigned nb_o_d) :
    bouton_plus({at[0] + 3*dimensions[0]/5, at[1]}, {dimensions[0]/5, dimensions[1]}, Button::etat::Inactive),
    bouton_moins(at, {dimensions[0]/5, dimensions[1]}, Button::etat::Inactive),
    bouton_envoyer({at[0] + 4*dimensions[0]/5, at[1]}, {dimensions[0]/5, dimensions[1]}, Button::etat::Inactive),
    bouton_renvoyer({at[0] + 4*dimensions[0]/5, at[1] - dimensions[1]}, {dimensions[0]/5, dimensions[1]}, Button::etat::Inactive),
    zone{at, at + dimensions},
    nb_ouvriers_defaut{nb_o_d}, nb_ouvriers(nb_o_d), nb_ouvriers_envoyes{0} {}

void affichage::ButtonsOuvriers::render(sdl2::window & fenetre)
{
    auto dim_el = bouton_moins.getZone();
    auto dimy = zone[1][1] - zone[0][1];
    sdl2::font fnt("./data/Lato-Medium.ttf", static_cast<int>(dimy/1.35));
    sdl2::font fnt_info("./data/Lato-Medium.ttf", static_cast<int>(dimy/1.8));
    sdl2::texte txt(std::to_string(nb_ouvriers), fnt, fenetre, 2*(dim_el[1] - dim_el[0])[0], {255, 255, 255, 255}, {0, 0, 0, 0}, 1);
    sdl2::texte infos(std::to_string(nb_ouvriers_envoyes) + " ouvrier(s) ici", fnt_info, fenetre, 4*(dim_el[1] - dim_el[0])[0], {255, 255, 255, 255}, {0, 0, 0, 0}, 0);
    txt.at(dim_el[1][0], dim_el[0][1] + 1);
    infos.at(dim_el[0][0], dim_el[0][1] - dimy);

    fenetre << bouton_moins << bouton_plus << bouton_envoyer << bouton_renvoyer << txt << infos << sdl2::flush;
}

unsigned & affichage::ButtonsOuvriers::setNb_ouvriers(unsigned nb)
{
    if (nb > nb_ouvriers_disponibles) return nb_ouvriers = nb_ouvriers_disponibles;
    else return nb_ouvriers = nb;
}

unsigned & affichage::ButtonsOuvriers::setNb_ouvriers_envoyes(unsigned nbe)
{
    return nb_ouvriers_envoyes = nbe;
}

affichage::ButtonsOuvriers & affichage::ButtonsOuvriers::rapatrier()
{
    nb_ouvriers = nb_ouvriers_defaut;
    nb_ouvriers_envoyes = 0;
    return *this;
}

affichage::ButtonsOuvriers & affichage::ButtonsOuvriers::envoyer()
{
    setNb_ouvriers_envoyes(nb_ouvriers_envoyes + nb_ouvriers);
    nb_ouvriers = 0;
    return *this;
}

affichage::ButtonsOuvriers & affichage::ButtonsOuvriers::operator++()
{
    if (nb_ouvriers == nb_ouvriers_disponibles)
    {
        setNb_ouvriers(0);
        return *this;
    }
    setNb_ouvriers(++nb_ouvriers);
    return *this;
}
affichage::ButtonsOuvriers & affichage::ButtonsOuvriers::operator--()
{
    setNb_ouvriers(--nb_ouvriers);
    return *this;
}
affichage::ButtonsOuvriers::ButtonMore & affichage::ButtonsOuvriers::getBouton_plus()
{
    return bouton_plus;
}
affichage::ButtonsOuvriers::ButtonLess & affichage::ButtonsOuvriers::getBouton_moins()
{
    return bouton_moins;
}
affichage::ButtonsOuvriers::ButtonGo & affichage::ButtonsOuvriers::getBouton_envoyer()
{
    return bouton_envoyer;
}
affichage::ButtonsOuvriers::ButtonUndo & affichage::ButtonsOuvriers::getBouton_renvoyer()
{
    return bouton_renvoyer;
}

void affichage::ButtonsOuvriers::setEtats(std::array<affichage::Button::etat, 4ULL> Etats)
{
    bouton_plus.setEtat(Etats[0]);
    bouton_moins.setEtat(Etats[1]);
    bouton_envoyer.setEtat(Etats[2]);
    bouton_renvoyer.setEtat(Etats[3]);

}

void Jeu::update(double production_actuelle, std::size_t score_actuel)
{
    production_cum += production_actuelle;
    score = score_actuel;
    ++nombre_de_tours;
}

unsigned Jeu::getNombre_de_tours()
{
    return nombre_de_tours;
}

double Jeu::getProduction_moyenne(){return (nombre_de_tours != 0) ? production_cum/nombre_de_tours : production_cum;}

affichage::modes & Jeu::setMode(affichage::modes M)
{
    return Mode = M;
}

affichage::modes Jeu::getMode()
{
    return Mode;
}

std::size_t Jeu::getScore()
{
    return score;
}



// Début des fonction d'affichage des différents écrans

bool affichage_radioprotection(sdl2::window & fenetre, Centrale & centrale, Jeu & jeu, Dispatching & dispatching, unsigned delta_tour_ms) noexcept
{
    using namespace affichage;

    auto [DIMx, DIMy] = fenetre.dimensions();

    affichage::point<int> DIM_button = {DIMx/10, DIMy/35};

    sdl2::font fonte_soustitre("./data/Lato-Bold.ttf", 10*DIMx/1000);
    sdl2::font fonte_titre("./data/Lato-Bold.ttf", 25*DIMx/1000);
    BarreEtat Etat_injecteur_acide_borique("État injecteur acide borique", 0, 1, 1, {DIMx/10, DIMy/5}, {DIMx/10, DIMy/50});
    BarreEtat Etat_pressuriseur("État pressuriseur", 0, 1, 1, {DIMx/10, static_cast<int>(DIMy/3.33)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_Pompe_circuit_primaire("État pompe circuit primaire", 0, 1, 1, {DIMx/10, static_cast<int>(DIMy/2.5)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_generateur_de_vapeur("État génerateur de vapeur", 0, 1, 1, {DIMx/10, DIMy/2}, {DIMx/10, DIMy/50});
    BarreEtat Etat_Pompe_circuit_secondaire("État pompe circuit secondaire", 0, 1, 1, {DIMx/10, static_cast<int>(DIMy/1.67)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_circuit_secondaire("État circuit secondaire", 0, 1, 1, {DIMx/10, static_cast<int>(DIMy/1.43)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_condenseur("État condenseur", 0, 1, 1, {DIMx/10, static_cast<int>(DIMy/1.25)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_circuit_primaire("État circuit primaire", 0, 1, 1, {DIMx/10, static_cast<int>(DIMy/1.11)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_Barre_de_graphite("État barre de graphite", 0, 1, 1, {4*DIMx/10, DIMy/5}, {DIMx/10, DIMy/50});
    BarreEtat Etat_canaux("État canaux", 0, 1, 1, {4*DIMx/10, static_cast<int>(DIMy/3.33)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_piscine("État piscine", 0, 1, 1, {4*DIMx/10, static_cast<int>(DIMy/2.5)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_Cuve("État Cuve", 0, 1, 1, {4*DIMx/10, DIMy/2}, {DIMx/10, DIMy/50});
    BarreEtat Etat_echangeur_de_chaleur("État échangeur de chaleur", 0, 1, 1, {4*DIMx/10, static_cast<int>(DIMy/1.67)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_enceinte_de_confinement("État enceinte de confinement", 0, 1, 1, {4*DIMx/10, static_cast<int>(DIMy/1.43)}, {DIMx/10, DIMy/50});
    BarreEtat Etat_centrale("État centrale", 0, 1, 1, {4*DIMx/10, static_cast<int>(DIMy/1.25)}, {DIMx/10, DIMy/50});

    Timer Tour({DIMx/50, DIMy/50}, {DIMx/20, DIMy/100});
    Tour.setNombre_de_tours(jeu.getNombre_de_tours());

    ButtonsOuvriers B_Injecteur_acide_borique({static_cast<int>(DIMx/4), static_cast<int>(DIMy/5)}, DIM_button, 5);
    ButtonsOuvriers B_Pressuriseur({static_cast<int>(DIMx/4), static_cast<int>(DIMy/3.33)}, DIM_button, 10);
    ButtonsOuvriers B_Pompe_circuit_primaire({static_cast<int>(DIMx/4), static_cast<int>(DIMy/2.5)}, DIM_button, 8);
    ButtonsOuvriers B_Generateur_de_vapeur({static_cast<int>(DIMx/4), static_cast<int>(DIMy/2)}, DIM_button, 60);
    ButtonsOuvriers B_Pompe_circuit_secondaire({static_cast<int>(DIMx/4), static_cast<int>(DIMy/1.67)}, DIM_button, 8);
    ButtonsOuvriers B_Circuit_secondaire({static_cast<int>(DIMx/4), static_cast<int>(DIMy/1.43)}, DIM_button, 14);
    ButtonsOuvriers B_Condenseur({static_cast<int>(DIMx/4), static_cast<int>(DIMy/1.25)}, DIM_button, 20);
    ButtonsOuvriers B_Circuit_primaire({static_cast<int>(DIMx/4), static_cast<int>(DIMy/1.11)}, DIM_button, 45);

    sdl2::texte titre1("POSTE DE SÉCURITÉ RADIOPROTECTION", fonte_titre, fenetre, {0xFF,0xFF,0xFF,0xFF});

    titre1.at(static_cast<int>(DIMx/11.54), static_cast<int>(DIMx/25));

    sdl2::texte titre3("Appuyer sur T pour repasser en salle de controle", fonte_soustitre, fenetre, {0xFF,0xFF,0xFF,0xFF});

    titre3.at(static_cast<int>(DIMx/1.44), static_cast<int>(DIMy/1.04));


    /*
        Ici on choisit de donner la possibilité au joueur de prolonger son tour s'il veut envoyer des ouvriers
        out tout simplement de regarder les états des différents organes.
    */
    sdl2::event_queue queue;
    std::chrono::steady_clock::time_point temps_depart, temps_arrivee;
    unsigned delta_temps = 0;

    bool ButtonEmergency_pressed = false;
    bool ButtonPop_pressed = false;

    std::string Identifiant;
    ButtonStop ButtonEmergency({11*DIMx/15, 3*DIMy/4}, {2*DIMx/15, DIMy/10});
    ButtonPopulation ButtonPop({11*DIMx/15, 5*DIMy/8}, {2*DIMx/15, DIMy/10});

    // Création du conteneur des hitbox des différentes zones interactives avec la souris

    CurrentRenders Cur(paire("BE", {ButtonEmergency.getZone()[0], ButtonEmergency.getZone()[1] - ButtonEmergency.getZone()[0]})
    , paire("BP", {ButtonPop.getZone()[0], ButtonPop.getZone()[1] - ButtonPop.getZone()[0]})
    , paire("+Iab", {B_Injecteur_acide_borique.getBouton_plus().getPosition(), B_Injecteur_acide_borique.getBouton_plus().dimensions()})
    , paire("-Iab", {B_Injecteur_acide_borique.getBouton_moins().getPosition(), B_Injecteur_acide_borique.getBouton_moins().dimensions()})
    , paire("GIab", {B_Injecteur_acide_borique.getBouton_envoyer().getPosition(), B_Injecteur_acide_borique.getBouton_envoyer().dimensions()})
    , paire("UIab", {B_Injecteur_acide_borique.getBouton_renvoyer().getPosition(), B_Injecteur_acide_borique.getBouton_renvoyer().dimensions()})
    , paire("+P", {B_Pressuriseur.getBouton_plus().getPosition(), B_Pressuriseur.getBouton_plus().dimensions()})
    , paire("-P", {B_Pressuriseur.getBouton_moins().getPosition(), B_Pressuriseur.getBouton_moins().dimensions()})
    , paire("GP", {B_Pressuriseur.getBouton_envoyer().getPosition(), B_Pressuriseur.getBouton_envoyer().dimensions()})
    , paire("UP", {B_Pressuriseur.getBouton_renvoyer().getPosition(), B_Pressuriseur.getBouton_renvoyer().dimensions()})
    , paire("+Pcp", {B_Pompe_circuit_primaire.getBouton_plus().getPosition(), B_Pompe_circuit_primaire.getBouton_plus().dimensions()})
    , paire("-Pcp", {B_Pompe_circuit_primaire.getBouton_moins().getPosition(), B_Pompe_circuit_primaire.getBouton_moins().dimensions()})
    , paire("GPcp", {B_Pompe_circuit_primaire.getBouton_envoyer().getPosition(), B_Pompe_circuit_primaire.getBouton_envoyer().dimensions()})
    , paire("UPcp", {B_Pompe_circuit_primaire.getBouton_renvoyer().getPosition(), B_Pompe_circuit_primaire.getBouton_renvoyer().dimensions()})
    , paire("+Gdv", {B_Generateur_de_vapeur.getBouton_plus().getPosition(), B_Generateur_de_vapeur.getBouton_plus().dimensions()})
    , paire("-Gdv", {B_Generateur_de_vapeur.getBouton_moins().getPosition(), B_Generateur_de_vapeur.getBouton_moins().dimensions()})
    , paire("GGdv", {B_Generateur_de_vapeur.getBouton_envoyer().getPosition(), B_Generateur_de_vapeur.getBouton_renvoyer().dimensions()})
    , paire("UGdv", {B_Generateur_de_vapeur.getBouton_renvoyer().getPosition(), B_Generateur_de_vapeur.getBouton_envoyer().dimensions()})
    , paire("+Pcs", {B_Pompe_circuit_secondaire.getBouton_plus().getPosition(), B_Pompe_circuit_secondaire.getBouton_plus().dimensions()})
    , paire("-Pcs", {B_Pompe_circuit_secondaire.getBouton_moins().getPosition(), B_Pompe_circuit_secondaire.getBouton_moins().dimensions()})
    , paire("GPcs", {B_Pompe_circuit_secondaire.getBouton_envoyer().getPosition(), B_Pompe_circuit_secondaire.getBouton_envoyer().dimensions()})
    , paire("UPcs", {B_Pompe_circuit_secondaire.getBouton_renvoyer().getPosition(), B_Pompe_circuit_secondaire.getBouton_renvoyer().dimensions()})
    , paire("+Cs", {B_Circuit_secondaire.getBouton_plus().getPosition(), B_Circuit_secondaire.getBouton_plus().dimensions()})
    , paire("-Cs", {B_Circuit_secondaire.getBouton_moins().getPosition(), B_Circuit_secondaire.getBouton_moins().dimensions()})
    , paire("GCs", {B_Circuit_secondaire.getBouton_envoyer().getPosition(), B_Circuit_secondaire.getBouton_envoyer().dimensions()})
    , paire("UCs", {B_Circuit_secondaire.getBouton_renvoyer().getPosition(), B_Circuit_secondaire.getBouton_renvoyer().dimensions()})
    , paire("+C", {B_Condenseur.getBouton_plus().getPosition(), B_Condenseur.getBouton_plus().dimensions()})
    , paire("-C", {B_Condenseur.getBouton_moins().getPosition(), B_Condenseur.getBouton_moins().dimensions()})
    , paire("GC", {B_Condenseur.getBouton_envoyer().getPosition(), B_Condenseur.getBouton_envoyer().dimensions()})
    , paire("UC", {B_Condenseur.getBouton_renvoyer().getPosition(), B_Condenseur.getBouton_renvoyer().dimensions()})
    , paire("+Cp", {B_Circuit_primaire.getBouton_plus().getPosition(), B_Circuit_primaire.getBouton_plus().dimensions()})
    , paire("-Cp", {B_Circuit_primaire.getBouton_moins().getPosition(), B_Circuit_primaire.getBouton_moins().dimensions()})
    , paire("GCp", {B_Circuit_primaire.getBouton_envoyer().getPosition(), B_Circuit_primaire.getBouton_envoyer().dimensions()})
    , paire("UCp", {B_Circuit_primaire.getBouton_renvoyer().getPosition(), B_Circuit_primaire.getBouton_renvoyer().dimensions()}));

    B_Injecteur_acide_borique.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Injecteur_acide_borique]);
    B_Pressuriseur.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Pressuriseur]);
    B_Pompe_circuit_primaire.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Pompe_circuit_primaire]);
    B_Generateur_de_vapeur.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Generateur_de_vapeur]);
    B_Pompe_circuit_secondaire.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Pompe_circuit_secondaire]);
    B_Circuit_secondaire.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Circuit_secondaire]);
    B_Condenseur.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Condenseur]);
    B_Circuit_primaire.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Circuit_primaire]);

    map_etats_t etats_ouvriers;

    for(;;)
    {
        if (centrale.est_arrete())
        {
            ButtonEmergency.setEtat(Button::etat::Active);
            fenetre << ButtonEmergency << sdl2::flush;
        }
        if (centrale.est_evacuee())
        {
            ButtonPop.setEtat(Button::etat::Active);
            fenetre << ButtonPop << sdl2::flush;
        }

        Etat_injecteur_acide_borique.setValeurActuelle(centrale.reacteur.E_bore());
        Etat_pressuriseur.setValeurActuelle(centrale.circuit_primaire.E_press());
        Etat_Pompe_circuit_primaire.setValeurActuelle(centrale.circuit_primaire.E_p1());
        Etat_generateur_de_vapeur.setValeurActuelle(centrale.circuit_secondaire.E_vap());
        Etat_Pompe_circuit_secondaire.setValeurActuelle(centrale.circuit_secondaire.E_p2());
        Etat_circuit_secondaire.setValeurActuelle(centrale.circuit_secondaire.E_C2());
        Etat_condenseur.setValeurActuelle(centrale.circuit_secondaire.E_cd());
        Etat_circuit_primaire.setValeurActuelle(centrale.circuit_primaire.E_C1());
        Etat_Barre_de_graphite.setValeurActuelle(centrale.reacteur.E_barre());
        Etat_canaux.setValeurActuelle(centrale.reacteur.E_canaux());
        Etat_piscine.setValeurActuelle(centrale.reacteur.E_pisc());
        Etat_Cuve.setValeurActuelle(centrale.reacteur.E_cuve());
        Etat_echangeur_de_chaleur.setValeurActuelle(centrale.circuit_primaire.E_EC());
        Etat_enceinte_de_confinement.setValeurActuelle(centrale.E_enceinte());
        Etat_centrale.setValeurActuelle(centrale.E_centrale());

        etats_ouvriers = centrale.etats_ouvriers();
        if (jeu.getMode() != affichage::modes::Mode1)
        {
            sdl2::texte score("Score : " + std::to_string(dispatching.score()), fonte_soustitre, fenetre, {255, 255, 255, 0});
            score.at(static_cast<int>(DIMx/1.44), static_cast<int>(4*DIMy/30));
            sdl2::rectangle rect_score(score.getPosition(), score.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
            fenetre << rect_score << score;
        }
        sdl2::texte val_rad1("Radioactivité circuit primaire : " + std::to_string(static_cast<int>(centrale.circuit_primaire.R1())) + " mrad", fonte_soustitre, fenetre, {255, 255, 255, 0});
        sdl2::texte val_rad2("Radioactivité circuit secondaire : " + std::to_string(static_cast<int>(centrale.circuit_secondaire.R2())) + " mrad", fonte_soustitre, fenetre, {255, 255, 255, 0});
        sdl2::texte val_rad3("Radioactivité de l'enceinte de confinement : " + std::to_string(static_cast<int>(centrale.R_enceinte())) + " mrad", fonte_soustitre, fenetre, {255, 255, 255, 0});
        sdl2::texte val_rad4("Radioactivité de l'air : " + std::to_string(static_cast<int>(centrale.population.R_air())) + " mrad", fonte_soustitre, fenetre, {255, 255, 255, 0});
        sdl2::texte val_rad5("Radioactivité de l'eau : " + std::to_string(static_cast<int>(centrale.population.R_eau())) + " mrad", fonte_soustitre, fenetre, {255, 255, 255, 0});
        sdl2::texte val_rad6("Radioactivité de la piscine : " + std::to_string(static_cast<int>(centrale.reacteur.R_pisc())) + " mrad", fonte_soustitre, fenetre, {255, 255, 255, 0});
        sdl2::texte val_pop1("Taux d'évacuation : " + std::to_string(static_cast<int>(centrale.population.evac()*100)) + " %", fonte_soustitre, fenetre, {255, 255, 255, 0});
        sdl2::texte val_pop2("Contamination : " + std::to_string(centrale.population.contamination()), fonte_soustitre, fenetre, {255, 255, 255, 0});

        sdl2::texte ouvriers_malades("Ouvriers malades : " + std::to_string(etats_ouvriers[Etat::Malade]), fonte_soustitre, fenetre, {0xFF,0xFF,0xFF,0xFF});
        sdl2::texte ouvriers_actifs("Ouvriers actifs : " + std::to_string(etats_ouvriers[Etat::Actif]), fonte_soustitre, fenetre, {0xFF,0xFF,0xFF,0xFF});
        sdl2::texte ouvriers_inactifs("Ouvriers inactifs : " + std::to_string(etats_ouvriers[Etat::Inactif]), fonte_soustitre, fenetre, {0xFF,0xFF,0xFF,0xFF});


        val_rad1.at(static_cast<int>(DIMx/1.44), static_cast<int>(6*DIMy/30));
        val_rad2.at(static_cast<int>(DIMx/1.44), static_cast<int>(7*DIMy/30));
        val_rad3.at(static_cast<int>(DIMx/1.44), static_cast<int>(8*DIMy/30));
        val_rad4.at(static_cast<int>(DIMx/1.44), static_cast<int>(9*DIMy/30));
        val_rad5.at(static_cast<int>(DIMx/1.44), static_cast<int>(10*DIMy/30));
        val_rad6.at(static_cast<int>(DIMx/1.44), static_cast<int>(11*DIMy/30));
        val_pop1.at(static_cast<int>(DIMx/1.44), static_cast<int>(12*DIMy/30));
        val_pop2.at(static_cast<int>(DIMx/1.44), static_cast<int>(13*DIMy/30));

        ouvriers_actifs.at(static_cast<int>(DIMx/1.44), static_cast<int>(14*DIMy/30));
        ouvriers_malades.at(static_cast<int>(DIMx/1.44), static_cast<int>(15*DIMy/30));
        ouvriers_inactifs.at(static_cast<int>(DIMx/1.44), static_cast<int>(16*DIMy/30));

        sdl2::rectangle rect_rad1(val_rad1.getPosition(), val_rad1.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
        sdl2::rectangle rect_rad2(val_rad2.getPosition(), val_rad2.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
        sdl2::rectangle rect_rad3(val_rad3.getPosition(), val_rad3.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
        sdl2::rectangle rect_rad4(val_rad4.getPosition(), val_rad4.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
        sdl2::rectangle rect_rad5(val_rad5.getPosition(), val_rad5.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
        sdl2::rectangle rect_rad6(val_rad6.getPosition(), val_rad6.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
        sdl2::rectangle rect_pop1(val_pop1.getPosition(), val_pop1.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
        sdl2::rectangle rect_pop2(val_pop2.getPosition(), val_pop2.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);

        sdl2::rectangle rect_ouvriers_malades(ouvriers_malades.getPosition(), ouvriers_malades.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
        sdl2::rectangle rect_ouvriers_actifs(ouvriers_actifs.getPosition(), ouvriers_actifs.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);
        sdl2::rectangle rect_ouvriers_inactifs(ouvriers_inactifs.getPosition(), ouvriers_inactifs.dimension()*std::array<int, 2ULL>{2, 1}, {0, 0, 0, 0}, true);


        // Gestion de la possibilité d'envoi d'ouvriers via le bouton "GO"

        map_reparables_t organes_reparables = centrale.organes_reparables();

        B_Injecteur_acide_borique.getBouton_envoyer().setEtat((organes_reparables[Organe::Injecteur_acide_borique] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive);
        B_Pressuriseur.getBouton_envoyer().setEtat((organes_reparables[Organe::Pressuriseur] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive);
        B_Pompe_circuit_primaire.getBouton_envoyer().setEtat((organes_reparables[Organe::Pompe_circuit_primaire] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive);
        B_Generateur_de_vapeur.getBouton_envoyer().setEtat((organes_reparables[Organe::Generateur_de_vapeur] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive);
        B_Pompe_circuit_secondaire.getBouton_envoyer().setEtat((organes_reparables[Organe::Pompe_circuit_secondaire] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive);
        B_Circuit_secondaire.getBouton_envoyer().setEtat((organes_reparables[Organe::Circuit_secondaire] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive);
        B_Condenseur.getBouton_envoyer().setEtat((organes_reparables[Organe::Condenseur] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive);
        B_Circuit_primaire.getBouton_envoyer().setEtat((organes_reparables[Organe::Circuit_primaire] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive);
        // Mise à jour des éléments de la fenêtre

        fenetre << titre1 << titre3 << Etat_injecteur_acide_borique << Etat_pressuriseur
        << Etat_Pompe_circuit_primaire << Etat_generateur_de_vapeur << Etat_Pompe_circuit_secondaire
        << Etat_circuit_secondaire << Etat_condenseur << Etat_circuit_primaire << Etat_Barre_de_graphite
        << Etat_canaux << Etat_piscine << Etat_Cuve << Etat_echangeur_de_chaleur << Etat_enceinte_de_confinement
        << Etat_centrale << ButtonEmergency << ButtonPop << B_Injecteur_acide_borique
        << B_Pressuriseur << B_Pompe_circuit_primaire << B_Generateur_de_vapeur << B_Pompe_circuit_secondaire
        << B_Circuit_secondaire << B_Condenseur << B_Circuit_primaire << rect_rad1 << rect_rad2 << val_rad1 << val_rad2 << rect_ouvriers_malades << rect_ouvriers_actifs
        << rect_ouvriers_inactifs << ouvriers_malades << ouvriers_actifs
        << ouvriers_inactifs << rect_rad3 << rect_rad4 << rect_rad5 << rect_rad6 << rect_pop1 << rect_pop2
        << val_rad3 << val_rad4 << val_rad5 << val_rad6 << val_pop1 << val_pop2 << Tour << sdl2::flush;

        delta_temps = 0;
        temps_depart = std::chrono::steady_clock::now();

        while (delta_temps < delta_tour_ms)
        {
            auto events = queue.pull_events();
            for (const auto& e : events)
            {
                if (e->kind_of_event() == sdl2::event::quit)
                {
                    jeu.setMode(affichage::modes::Void);
                    return true;
                }
                if (e->kind_of_event() == sdl2::event::mouse_motion)
                {
                    auto& key_ev = dynamic_cast<sdl2::event_mouse_motion&>(*e);
                    Identifiant = Cur.where(key_ev.position());
                }
                if (e->kind_of_event() == sdl2::event::mouse_button)
                {
                    auto& key_ev = dynamic_cast<sdl2::event_mouse_button&>(*e);
                    if (key_ev.button() == sdl2::event_mouse_button::left)
                    {
                        if (key_ev.state() == sdl2::event_mouse_button::pressed)
                        {
                            if (Identifiant == "BE" && !centrale.est_arrete())
                            {
                                ButtonEmergency.setEtat(Button::etat::Enfonce);
                                fenetre << ButtonEmergency << sdl2::flush;
                                ButtonEmergency_pressed = true;
                            }
                            if (Identifiant == "BP" && !centrale.est_evacuee())
                            {
                                ButtonPop.setEtat(Button::etat::Enfonce);
                                fenetre << ButtonPop << sdl2::flush;
                                ButtonPop_pressed = true;
                            }

                            /*
                                Evenements sur la gestions des ouvriers via les boutons +/-/GO/UNDO
                            */
                            if (Identifiant == "+Iab")
                            {
                                B_Injecteur_acide_borique.getBouton_plus().setEtat(Button::etat::Enfonce);
                                ++B_Injecteur_acide_borique;
                                fenetre << B_Injecteur_acide_borique << sdl2::flush;
                            }
                            if (Identifiant == "-Iab")
                            {
                                B_Injecteur_acide_borique.getBouton_moins().setEtat(Button::etat::Enfonce);
                                --B_Injecteur_acide_borique;
                                fenetre << B_Injecteur_acide_borique << sdl2::flush;
                            }
                            if (Identifiant == "GIab" && B_Injecteur_acide_borique.getBouton_envoyer().getEtat() == affichage::Button::etat::Active)
                            {
                                centrale.affecter_ouvriers(Organe::Injecteur_acide_borique, B_Injecteur_acide_borique.getNb_ouvriers());
                                B_Injecteur_acide_borique.getBouton_envoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Injecteur_acide_borique << sdl2::flush;

                            }
                            if (Identifiant == "UIab")
                            {
                                centrale.desaffecter_ouvriers(Organe::Injecteur_acide_borique);
                                B_Injecteur_acide_borique.getBouton_renvoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Injecteur_acide_borique << sdl2::flush;
                            }

                            if (Identifiant == "+P")
                            {
                                B_Pressuriseur.getBouton_plus().setEtat(Button::etat::Enfonce);
                                ++B_Pressuriseur;
                                fenetre << B_Pressuriseur << sdl2::flush;
                            }
                            if (Identifiant == "-P")
                            {
                                B_Pressuriseur.getBouton_moins().setEtat(Button::etat::Enfonce);
                                --B_Pressuriseur;
                                fenetre << B_Pressuriseur << sdl2::flush;
                            }
                            if (Identifiant == "GP" && B_Pressuriseur.getBouton_envoyer().getEtat() == affichage::Button::etat::Active)
                            {
                                centrale.affecter_ouvriers(Organe::Pressuriseur, B_Pressuriseur.getNb_ouvriers());
                                B_Pressuriseur.getBouton_envoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Pressuriseur << sdl2::flush;
                            }
                            if (Identifiant == "UP")
                            {
                                centrale.desaffecter_ouvriers(Organe::Pressuriseur);
                                B_Pressuriseur.getBouton_renvoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Pressuriseur << sdl2::flush;
                            }

                            if (Identifiant == "+Pcp")
                            {
                                B_Pompe_circuit_primaire.getBouton_plus().setEtat(Button::etat::Enfonce);
                                ++B_Pompe_circuit_primaire;
                                fenetre << B_Pompe_circuit_primaire << sdl2::flush;
                            }
                            if (Identifiant == "-Pcp")
                            {
                                B_Pompe_circuit_primaire.getBouton_moins().setEtat(Button::etat::Enfonce);
                                --B_Pompe_circuit_primaire;
                                fenetre << B_Pompe_circuit_primaire << sdl2::flush;
                            }
                            if (Identifiant == "GPcp" && B_Pompe_circuit_primaire.getBouton_envoyer().getEtat() == affichage::Button::etat::Active)
                            {
                                centrale.affecter_ouvriers(Organe::Pompe_circuit_primaire, B_Pompe_circuit_primaire.getNb_ouvriers());
                                B_Pompe_circuit_primaire.getBouton_envoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Pompe_circuit_primaire << sdl2::flush;
                            }
                            if (Identifiant == "UPcp")
                            {
                                centrale.desaffecter_ouvriers(Organe::Pompe_circuit_primaire);
                                B_Pompe_circuit_primaire.getBouton_renvoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Pompe_circuit_primaire << sdl2::flush;
                            }

                            if (Identifiant == "+Gdv")
                            {
                                B_Generateur_de_vapeur.getBouton_plus().setEtat(Button::etat::Enfonce);
                                ++B_Generateur_de_vapeur;
                                fenetre << B_Generateur_de_vapeur << sdl2::flush;
                            }
                            if (Identifiant == "-Gdv")
                            {
                                B_Generateur_de_vapeur.getBouton_moins().setEtat(Button::etat::Enfonce);
                                --B_Generateur_de_vapeur;
                                fenetre << B_Generateur_de_vapeur << sdl2::flush;
                            }
                            if (Identifiant == "GGdv" && B_Generateur_de_vapeur.getBouton_envoyer().getEtat() == affichage::Button::etat::Active)
                            {
                                centrale.affecter_ouvriers(Organe::Generateur_de_vapeur, B_Generateur_de_vapeur.getNb_ouvriers());
                                B_Generateur_de_vapeur.getBouton_envoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Generateur_de_vapeur << sdl2::flush;
                            }
                            if (Identifiant == "UGdv")
                            {
                                centrale.desaffecter_ouvriers(Organe::Generateur_de_vapeur);
                                B_Generateur_de_vapeur.getBouton_renvoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Generateur_de_vapeur << sdl2::flush;
                            }

                            if (Identifiant == "+Pcs")
                            {
                                B_Pompe_circuit_secondaire.getBouton_plus().setEtat(Button::etat::Enfonce);
                                ++B_Pompe_circuit_secondaire;
                                fenetre << B_Pompe_circuit_secondaire << sdl2::flush;
                            }
                            if (Identifiant == "-Pcs")
                            {
                                B_Pompe_circuit_secondaire.getBouton_moins().setEtat(Button::etat::Enfonce);
                                --B_Pompe_circuit_secondaire;
                                fenetre << B_Pompe_circuit_secondaire << sdl2::flush;
                            }
                            if (Identifiant == "GPcs" && B_Pompe_circuit_secondaire.getBouton_envoyer().getEtat() == affichage::Button::etat::Active)
                            {
                                centrale.affecter_ouvriers(Organe::Pompe_circuit_secondaire, B_Pompe_circuit_secondaire.getNb_ouvriers());
                                B_Pompe_circuit_secondaire.getBouton_envoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Pompe_circuit_secondaire << sdl2::flush;
                            }
                            if (Identifiant == "UPcs")
                            {
                                centrale.desaffecter_ouvriers(Organe::Pompe_circuit_secondaire);
                                B_Pompe_circuit_secondaire.getBouton_renvoyer().setEtat(Button::etat::Enfonce);
                                fenetre << B_Pompe_circuit_secondaire << sdl2::flush;
                            }

                            if (Identifiant == "+Cs")
                            {
                                B_Circuit_secondaire.getBouton_plus().setEtat(Button::etat::Enfonce);
                                ++B_Circuit_secondaire;
                                fenetre << B_Circuit_secondaire << sdl2::flush;
                            }
                            if (Identifiant == "-Cs")
                            {
                                B_Circuit_secondaire.getBouton_moins().setEtat(affichage::Button::etat::Enfonce);
                                --B_Circuit_secondaire;
                                fenetre << B_Circuit_secondaire << sdl2::flush;
                            }
                            if (Identifiant == "GCs" && B_Circuit_secondaire.getBouton_envoyer().getEtat() == affichage::Button::etat::Active)
                            {
                                centrale.affecter_ouvriers(Organe::Circuit_secondaire, B_Circuit_secondaire.getNb_ouvriers());
                                B_Circuit_secondaire.getBouton_envoyer().setEtat(affichage::Button::etat::Enfonce);
                                fenetre << B_Circuit_secondaire << sdl2::flush;
                            }
                            if (Identifiant == "UCs")
                            {
                                centrale.desaffecter_ouvriers(Organe::Circuit_secondaire);
                                B_Circuit_secondaire.getBouton_renvoyer().setEtat(affichage::Button::etat::Enfonce);
                                fenetre << B_Circuit_secondaire << sdl2::flush;
                            }

                            if (Identifiant == "+C")
                            {
                                B_Condenseur.getBouton_plus().setEtat(affichage::Button::etat::Enfonce);
                                ++B_Condenseur;
                                fenetre << B_Condenseur << sdl2::flush;
                            }
                            if (Identifiant == "-C")
                            {
                                B_Condenseur.getBouton_moins().setEtat(affichage::Button::etat::Enfonce);
                                --B_Condenseur;
                                fenetre << B_Condenseur << sdl2::flush;
                            }
                            if (Identifiant == "GC" && B_Condenseur.getBouton_envoyer().getEtat() == affichage::Button::etat::Active)
                            {
                                centrale.affecter_ouvriers(Organe::Condenseur, B_Condenseur.getNb_ouvriers());
                                B_Condenseur.getBouton_envoyer().setEtat(affichage::Button::etat::Enfonce);
                                fenetre << B_Condenseur << sdl2::flush;
                            }
                            if (Identifiant == "UC")
                            {
                                centrale.desaffecter_ouvriers(Organe::Condenseur);
                                B_Condenseur.getBouton_renvoyer().setEtat(affichage::Button::etat::Enfonce);
                                fenetre << B_Condenseur << sdl2::flush;
                            }

                            if (Identifiant == "+Cp")
                            {
                                B_Circuit_primaire.getBouton_plus().setEtat(affichage::Button::etat::Enfonce);
                                ++B_Circuit_primaire;
                                fenetre << B_Circuit_primaire << sdl2::flush;
                            }
                            if (Identifiant == "-Cp")
                            {
                                B_Circuit_primaire.getBouton_moins().setEtat(affichage::Button::etat::Enfonce);
                                --B_Circuit_primaire;
                                fenetre << B_Circuit_primaire << sdl2::flush;
                            }
                            if (Identifiant == "GCp" && B_Circuit_primaire.getBouton_envoyer().getEtat() == affichage::Button::etat::Active)
                            {
                                centrale.affecter_ouvriers(Organe::Circuit_primaire, B_Circuit_primaire.getNb_ouvriers());
                                B_Circuit_primaire.getBouton_envoyer().setEtat(affichage::Button::etat::Enfonce);
                                fenetre << B_Circuit_primaire << sdl2::flush;
                            }
                            if (Identifiant == "UCp")
                            {
                                centrale.desaffecter_ouvriers(Organe::Circuit_primaire);
                                B_Circuit_primaire.getBouton_renvoyer().setEtat(affichage::Button::etat::Enfonce);
                                fenetre << B_Circuit_primaire << sdl2::flush;
                            }
                            B_Circuit_primaire.setEtats({affichage::Button::etat::Inactive, affichage::Button::etat::Inactive, (organes_reparables[Organe::Injecteur_acide_borique] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive, affichage::Button::etat::Inactive});
                            B_Condenseur.setEtats({affichage::Button::etat::Inactive, affichage::Button::etat::Inactive, (organes_reparables[Organe::Pressuriseur] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive, affichage::Button::etat::Inactive});
                            B_Circuit_secondaire.setEtats({affichage::Button::etat::Inactive, affichage::Button::etat::Inactive, (organes_reparables[Organe::Pompe_circuit_primaire] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive, affichage::Button::etat::Inactive});
                            B_Pompe_circuit_secondaire.setEtats({affichage::Button::etat::Inactive, affichage::Button::etat::Inactive, (organes_reparables[Organe::Generateur_de_vapeur] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive, affichage::Button::etat::Inactive});
                            B_Generateur_de_vapeur.setEtats({affichage::Button::etat::Inactive, affichage::Button::etat::Inactive, (organes_reparables[Organe::Pompe_circuit_secondaire] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive, affichage::Button::etat::Inactive});
                            B_Pompe_circuit_primaire.setEtats({affichage::Button::etat::Inactive, affichage::Button::etat::Inactive, (organes_reparables[Organe::Circuit_secondaire] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive, affichage::Button::etat::Inactive});
                            B_Pressuriseur.setEtats({affichage::Button::etat::Inactive, affichage::Button::etat::Inactive, (organes_reparables[Organe::Condenseur] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive, affichage::Button::etat::Inactive});
                            B_Injecteur_acide_borique.setEtats({affichage::Button::etat::Inactive, affichage::Button::etat::Inactive, (organes_reparables[Organe::Circuit_primaire] == true)? affichage::Button::etat::Active : affichage::Button::etat::Inactive, affichage::Button::etat::Inactive});

                            fenetre << B_Injecteur_acide_borique
                            << B_Pressuriseur << B_Pompe_circuit_primaire << B_Generateur_de_vapeur << B_Pompe_circuit_secondaire
                            << B_Circuit_secondaire << B_Condenseur << B_Circuit_primaire << sdl2::flush;

                            // Mise à jour du nombre d'ouvriers disponibles

                            B_Condenseur.nb_ouvriers_disponibles = centrale.etats_ouvriers()[Etat::Inactif];
                        }
                        else
                        {
                            if (ButtonPop_pressed == true)
                            {
                                ButtonPop.setEtat(affichage::Button::etat::Active);
                                fenetre << ButtonPop << sdl2::flush;
                            }
                        }
                    }
                }
                if (e->kind_of_event() == sdl2::event::key_down)
                {
                    auto & key_ev = dynamic_cast<sdl2::event_keyboard&>(*e);
                    if ((key_ev.code() == SDLK_t) && (key_ev.state() == sdl2::event_keyboard::pressed))
                    {
                        return false;
                    }
                    if ((key_ev.code() == SDLK_s) && (key_ev.state() == sdl2::event_keyboard::pressed))
                    {
                        jeu.fin_du_game = true;
                    }
                }
            }
            temps_arrivee = std::chrono::steady_clock::now();
            delta_temps = std::chrono::duration_cast<std::chrono::milliseconds>(temps_arrivee - temps_depart).count();
            Tour.setValeurActuelle(1 - delta_temps/static_cast<double>(delta_tour_ms));
            fenetre << Tour << sdl2::flush;
        }


        if (jeu.fin_du_game == true) break; // Instruction permettant au joueur de terminer sa session de jeu à tout moment

        centrale.update();
        centrale.reparation();
        centrale.alertes();

        switch (jeu.getMode())
        {
            case (affichage::modes::Mode1):
                jeu.update(centrale.MW());
                jeu.fin_du_game = (centrale.etat() == Urgence::DEFCOM_4);
                break;
            case (affichage::modes::Mode2):
                jeu.update(centrale.MW(), dispatching.score());          // Commande pour le mode dispatching
                jeu.fin_du_game = dispatching.update(centrale.MW()) && (centrale.etat() == Urgence::DEFCOM_4);
                break;
            case (affichage::modes::Mode3):
                jeu.update(centrale.MW(), dispatching.score());          // Commande pour le mode catastrophe et dispatching
                jeu.fin_du_game = dispatching.update(centrale.MW()) && (centrale.etat() == Urgence::DEFCOM_4);
                centrale.catastrophe();
                break;
            default:
                return true;
        }
        Tour.setNombre_de_tours(jeu.getNombre_de_tours());


        B_Injecteur_acide_borique.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Injecteur_acide_borique]);
        B_Pressuriseur.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Pressuriseur]);
        B_Pompe_circuit_primaire.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Pompe_circuit_primaire]);
        B_Generateur_de_vapeur.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Generateur_de_vapeur]);
        B_Pompe_circuit_secondaire.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Pompe_circuit_secondaire]);
        B_Circuit_secondaire.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Circuit_secondaire]);
        B_Condenseur.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Condenseur]);
        B_Circuit_primaire.setNb_ouvriers_envoyes(centrale.postes_ouvriers()[Organe::Circuit_primaire]);

        if (ButtonEmergency_pressed)
        {
            centrale.arret_urgence();
            ButtonEmergency_pressed = false;
            if (!centrale.est_arrete())
            {
                ButtonEmergency.setEtat(affichage::Button::etat::Inactive);
                fenetre << ButtonEmergency << sdl2::flush;
            }
        }
        if (ButtonPop_pressed)
        {
            centrale.evacuation_population();
            ButtonPop_pressed = false;
            if (!centrale.est_evacuee())
            {
                ButtonPop.setEtat(affichage::Button::etat::Inactive);
                fenetre << ButtonPop << sdl2::flush;
            }
        }
        SDL_Delay(17);
    }
    return true;    // Ligne pour faire plaisir au compilateur (étant donné qu'on ne sort jamais de cette boucle)
}


affichage::modes affichage_modes(sdl2::window & fenetre) noexcept
{
    using namespace affichage;

    int taille_simpson;
    auto [DIMx, DIMy] = fenetre.dimensions();   // 1000 et 700 en standard
    taille_simpson = DIMx/5;


    sdl2::font fonte_titre("./data/Lato-Bold.ttf", 50*DIMy/1000);
    sdl2::font fonte_modes("./data/Lato-Medium.ttf", 55*DIMy/1000);

    sdl2::image homer("data/homer.png", fenetre);
    homer.stretch({taille_simpson, taille_simpson});
    homer.at((DIMx-taille_simpson)/2, DIMy/10);

    sdl2::image burns("data/burns.png", fenetre);
    burns.stretch({taille_simpson, taille_simpson});
    burns.at((DIMx-taille_simpson)/2, DIMy/10);

    sdl2::image cata("data/burnsCata.png", fenetre);
    cata.stretch({taille_simpson, taille_simpson});
    cata.at((DIMx-taille_simpson)/2, DIMy/10);


    sdl2::texte titre("NUCLEAR ALERT", fonte_titre, fenetre, {0xFF,0xFF,0xFF,0xFF});
    auto [titre_x, titre_y] = titre.dimension();
    titre.at((DIMx-titre_x)/2, DIMy/40 - titre_y/2 + 10);
    rectangle Titre({titre.getPosition(), titre.dimension()});

    sdl2::texte mode1("Mode 1 : Joue librement", fonte_modes, fenetre, {0xFF,0xFF,0xFF,0xFF});
    auto [mode1_x, mode1_y] = mode1.dimension();
    mode1.at((DIMx - mode1_x)/2, DIMy/1.7 - mode1_y/2);
    rectangle Mode1({mode1.getPosition(), mode1.dimension()});

    sdl2::texte mode2("Mode 2 : Suis les conseils du dispatcher", fonte_modes, fenetre, {0xFF,0xFF,0xFF,0xFF});
    auto [mode2_x, mode2_y] = mode2.dimension();
    mode2.at((DIMx - mode2_x)/2, static_cast<int>(DIMy/1.7) + 2*mode1_y - mode2_y/2);
    rectangle Mode2({mode2.getPosition(), mode2.dimension()});

    sdl2::texte mode3("Mode 3 : Gère des catastrophes !", fonte_modes, fenetre, {0xFF,0xFF,0xFF,0xFF});
    auto [mode3_x, mode3_y] = mode3.dimension();
    mode3.at((DIMx - mode3_x)/2, DIMy/1.7 + 2*mode2_y + 2*mode1_y - mode3_y/2);
    rectangle Mode3({mode3.getPosition(), mode3.dimension()});


    bool quitting = false;

    std::string Identifiant;

    fenetre << titre << mode1 << mode2 << mode3 << sdl2::flush;
    CurrentRenders Curs(paire("titre", Titre), paire("mode1", Mode1), paire("mode2", Mode2), paire("mode3", Mode3));
    sdl2::event_queue queue;


    while (!quitting)
    {
        std::array<int, 2ULL> Position_curseur;
        auto events = queue.pull_events();
        for (const auto& e : events)
        {
            if (e->kind_of_event() == sdl2::event::quit)
            {
                quitting = true;
                break;
            }
            if (e->kind_of_event() == sdl2::event::mouse_motion)
            {
                auto& key_ev = dynamic_cast<sdl2::event_mouse_motion&>(*e);
                Position_curseur = key_ev.position();
                Identifiant = Curs.where(key_ev.position());

                if (Identifiant == "mode1")
                {
                    mode1.color_filter({17, 100, 249, 0});
                    fenetre << homer;
                }
                else if (Identifiant == "mode2")
                {
                    mode2.color_filter({17, 100, 249, 0});
                    fenetre << burns;
                }
                else if (Identifiant == "mode3")
                {
                    mode3.color_filter({17, 100, 249, 0});
                    fenetre << cata;
                }
                else
                {
                    mode1.color_filter({255, 255, 255, 0});
                    mode2.color_filter({255, 255, 255, 0});
                    mode3.color_filter({255, 255, 255, 0});
                }
                fenetre << titre << mode1 << mode2 << mode3 << sdl2::flush;
            }
            if (e->kind_of_event() == sdl2::event::mouse_button)
            {
                auto& key_ev = dynamic_cast<sdl2::event_mouse_button&>(*e);
                if ((key_ev.button() == sdl2::event_mouse_button::left) && (key_ev.state() == sdl2::event_mouse_button::pressed))
                {
                    if (Identifiant == "mode1")
                        return affichage::modes::Mode1;
                    if (Identifiant == "mode2")
                        return affichage::modes::Mode2;
                    if (Identifiant == "mode3")
                        return affichage::modes::Mode3;
                }
            }
            fenetre << titre << mode1 << mode2 << mode3 << sdl2::flush;
            SDL_Delay(17);
        }
    }
    return affichage::modes::Void;
}


affichage::modes affichage_salle_de_controle(sdl2::window & fenetre, Centrale & centrale, Jeu & jeu, unsigned delta_tour_ms = 1000) noexcept
{
    using namespace affichage;

    Dispatching dispatching;

    auto [DIMx, DIMy] = fenetre.dimensions();

    sdl2::font fonte_soustitre("./data/Lato-Bold.ttf", 10*DIMx/1000);
    sdl2::font fonte_titre("./data/Lato-Bold.ttf", 25*DIMx/1000);

    affichage::point<int> position_ordres{static_cast<int>(DIMx/2.17), static_cast<int>(DIMy/1.70)};
    affichage::point<int> position_catastrophes{static_cast<int>(DIMx/2.17), static_cast<int>(DIMy/1.93)};
    map_alertes_t Alertes;

    rectangle Zone = {point<int>({static_cast<int>(DIMx/2.15), static_cast<int>(DIMy/1.45)}), point<int>({static_cast<int>(DIMx/1.01), static_cast<int>(DIMy/1.02)})};

    // Création des barres non modifiables

    Barre val_debit_eau_1("débit eau circuit primaire", 0, 90, 0, {DIMx/10, static_cast<int>(DIMy/1.40)}, {DIMx/10, DIMy/50});
    Barre val_pression_1("pression circuit primaire", 1, 50, 1, {DIMx/10, static_cast<int>(DIMy/1.73)}, {DIMx/10, DIMy/50});
    Barre temperature_circuit_primaire("temperatre circuit primaire", 25, 600, 25, {DIMx/10, static_cast<int>(DIMy/1.55)}, {DIMx/10, DIMy/50});

    Barre val_pression_enceinte("valeur pression enceinte", 1, 50, 1, {DIMx/10, static_cast<int>(DIMy/3.33)}, {DIMx/10, DIMy/50});
    Barre valeur_debit_eau_3("Debit condenseur", 0, 150, 0, {static_cast<int>(DIMx/3.57), DIMy/5}, {DIMx/10, DIMy/50});
    Barre val_temp_vap("temperature vapeur", 25, 600, 25, {static_cast<int>(DIMx/3.57), static_cast<int>(DIMy/1.67)}, {DIMx/10, DIMy/50});
    Barre val_debit_eau_2("débit eau circuit secondaire", 0, 2, 0, {static_cast<int>(DIMx/3.57), static_cast<int>(DIMy/1.43)}, {DIMx/10, DIMy/50});
    Barre val_pression_2("pression vapeur", 1, 50, 1, {static_cast<int>(DIMx/3.57), static_cast<int>(DIMy/1.25)}, {DIMx/10, DIMy/50});
    Barre valeur_difference_de_temps("différence de température", 0, 20, 0, {static_cast<int>(DIMx/3.57), static_cast<int>(DIMy/3.33)}, {DIMx/10, DIMy/50});



    // Création des barres modifables
    BarreModifiable val_temperature_pressuriseur("température pressuriseur", 25, 200, 25, 25, 25, 100*(centrale.circuit_primaire.E_press() + centrale.circuit_primaire.E_res()), {DIMx/10, static_cast<int>(DIMy/1.25)}, {DIMx/10, DIMy/50});
    BarreModifiable val_pompe_1("regime pompe 1", 0, 1, 0, 0, 0, centrale.circuit_primaire.E_p1(), {DIMx/10, DIMy/2}, {DIMx/10, DIMy/50});

    BarreModifiable valeur_pompe_3("régime pompe condenseur", 0, 1, 0, 0, 0, centrale.circuit_secondaire.E_cd(), {DIMx/10,static_cast<int>(DIMy/2.5)}, {DIMx/10, DIMy/50});
    BarreModifiable taux_de_bore("taux de bore", 0, 0.5, centrale.reacteur.TB_eff(), centrale.reacteur.Tx_B(), 0, 0.5, {DIMx/10, DIMy/5}, {DIMx/10, DIMy/50});
    BarreModifiable val_pompe_2("regime pompe 2", 0, 1, 0, 0, 0, centrale.circuit_secondaire.E_p2(), {static_cast<int>(DIMx/3.57), DIMy/2}, {DIMx/10, DIMy/50});
    BarreModifiable valeur_etat_barre_graphite("proportion barres graphites", 0, 1, 0, 0, 1-centrale.reacteur.E_barre(), 1, {static_cast<int>(DIMx/3.57), static_cast<int>(DIMy/2.5)}, {DIMx/10, DIMy/50});

    Timer Tour({DIMx/50, DIMy/50}, {DIMx/20, DIMy/100});

    // Création des rectangles de {position, dimension} à donner en paramètre au constructeur de CurrentRenders

    rectangle Valeur_pompe_3 = {valeur_pompe_3.getZone()[0], valeur_pompe_3.getZone()[1] - valeur_pompe_3.getZone()[0]};
    rectangle Taux_de_bore = {taux_de_bore.getZone()[0], taux_de_bore.getZone()[1] - taux_de_bore.getZone()[0]};
    rectangle Val_pompe_1 = {val_pompe_1.getZone()[0], val_pompe_1.getZone()[1] - val_pompe_1.getZone()[0]};
    rectangle Val_temperature_pressuriseur = {val_temperature_pressuriseur.getZone()[0], val_temperature_pressuriseur.getZone()[1] - val_temperature_pressuriseur.getZone()[0]};
    rectangle Valeur_etat_barre_graphite = {valeur_etat_barre_graphite.getZone()[0], valeur_etat_barre_graphite.getZone()[1] - valeur_etat_barre_graphite.getZone()[0]};
    rectangle Val_pompe_2 = {val_pompe_2.getZone()[0], val_pompe_2.getZone()[1] - val_pompe_2.getZone()[0]};


    affichage::IllustrationCentrale image({false, false, false, false, false, false, false, false, false, false, false}, {static_cast<int>(DIMx/2.08), 0}, {static_cast<int>(DIMx/1.92), static_cast<int>(DIMy/2.17)});
    sdl2::rectangle CacheImg(image.getPosition(), image.dimension(), {0, 0, 0, 0}, true);

    sdl2::texte titre1("SALLE DE CONTROLE", fonte_titre, fenetre, {0xFF,0xFF,0xFF,0xFF});
    sdl2::texte titre2("DISPATCHING NATIONAL", fonte_soustitre, fenetre, {0xFF,0xFF,0xFF,0xFF});
    sdl2::texte titre_catastrophes("CATASTROPHES", fonte_soustitre, fenetre, {234, 85, 251});
    sdl2::texte titre3("ALERTE DEGRADATION", fonte_soustitre, fenetre, {0xFF,0xFF,0xFF,0xFF});
    sdl2::texte titre4("circuit pompe 1", fonte_soustitre, fenetre, {0xFF,0xFF,0xFF,0xFF});
    sdl2::texte titre5("circuit pompe 2", fonte_soustitre, fenetre, {0xFF,0xFF,0xFF,0xFF});

    titre1.at(static_cast<int>(DIMx/11.54), DIMx/25);
    titre_catastrophes.at(static_cast<int>(DIMx/2.17), static_cast<int>(DIMy/2.05));
    titre2.at(static_cast<int>(DIMx/2.17), static_cast<int>(DIMy/1.80));
    titre3.at(static_cast<int>(DIMx/2.17), static_cast<int>(DIMy/1.57));
    titre4.at(DIMx/10, static_cast<int>(DIMy/2.22));
    titre5.at(static_cast<int>(DIMx/3.57), static_cast<int>(DIMy/2.22));

    if (jeu.getMode() != modes::Mode1)
    {
        fenetre << titre2;
        if (jeu.getMode() == modes::Mode3)
            fenetre << titre_catastrophes;
    }

    fenetre << titre1 << titre3 << titre4 << titre5;
    std::string Identifiant;
    std::string message_catastrophe;
    std::string nouveau_message_catastrophe;

    double tol = 0.5;
    sdl2::event_queue queue;

    CurrentRenders Curs(paire("regime pompe condenseur", Valeur_pompe_3), paire("taux de bore", Taux_de_bore), paire("regime pompe 1", Val_pompe_1),
    paire("pressuriseur 1", Val_temperature_pressuriseur), paire("proportion barre graphite", Valeur_etat_barre_graphite),
    paire("regime pompe 2", Val_pompe_2));
    point<int> Position_curseur;
    std::chrono::steady_clock::time_point temps_depart, temps_arrivee;
    unsigned delta_temps;
    unsigned compteur_cata = 0;

    bool quitting = false;

    
    while (!quitting)   // Condition jamais respectée ici
    {

        // Mise à jour des éléments de la fenêtre
        PanneauAlertes Panneau(Alertes, "./data/Lato-Medium.ttf", Zone);

        sdl2::texte production("Production : " + std::to_string(static_cast<int>(centrale.MW())) + " MW", fonte_titre, fenetre, {0xFF,0xFF,0xFF,0xFF});

        production.at(static_cast<int>(DIMx/25), static_cast<int>(DIMy/1.1));

        sdl2::rectangle rect_prod(production.getPosition(), {static_cast<int>(production.dimension()[0]*1.5), production.dimension()[1]}, {0, 0, 0, 0}, true);

        Tour.setNombre_de_tours(jeu.getNombre_de_tours());

        fenetre << rect_prod << temperature_circuit_primaire << taux_de_bore << val_pression_enceinte <<
        valeur_pompe_3 << valeur_debit_eau_3 << val_pompe_1 << val_temp_vap << val_debit_eau_1 << val_pression_1 << Panneau << val_temperature_pressuriseur <<
        valeur_difference_de_temps << valeur_etat_barre_graphite << val_pompe_2 << val_debit_eau_2 << val_pression_2
        << production << Tour << sdl2::flush;

        delta_temps = 0;
        temps_depart = std::chrono::steady_clock::now();

        while (delta_temps < delta_tour_ms)
        {
            fenetre << taux_de_bore << val_pompe_1 << val_temperature_pressuriseur
            << valeur_etat_barre_graphite << val_pompe_2 << valeur_pompe_3 << sdl2::flush;

            auto events = queue.pull_events();

            for (const auto& e : events)
            {
                if (e->kind_of_event() == sdl2::event::quit)
                {
                    jeu.setMode(modes::Void);
                    return modes::Void;
                }
                if (e->kind_of_event() == sdl2::event::mouse_motion)
                {
                    auto& key_ev = dynamic_cast<sdl2::event_mouse_motion&>(*e);
                    Position_curseur = key_ev.position();

                }
                if (e->kind_of_event() == sdl2::event::mouse_button)
                {
                    auto & key_ev = dynamic_cast<sdl2::event_mouse_button&>(*e);
                    if ((key_ev.button() == sdl2::event_mouse_button::left) && (key_ev.state() == sdl2::event_mouse_button::pressed))
                    {
                        Identifiant = Curs.where(key_ev.position());
                        if (Identifiant == "taux de bore")
                            taux_de_bore.setValeurDemandee(Position_curseur);
                        if (Identifiant == "regime pompe 1")
                            val_pompe_1.setValeurDemandee(Position_curseur);
                        if (Identifiant == "pressuriseur 1")
                            val_temperature_pressuriseur.setValeurDemandee(Position_curseur);
                        if (Identifiant == "proportion barre graphite")
                            valeur_etat_barre_graphite.setValeurDemandee(Position_curseur);
                        if (Identifiant == "regime pompe 2")
                            val_pompe_2.setValeurDemandee(Position_curseur);
                        if (Identifiant == "regime pompe condenseur")
                            valeur_pompe_3.setValeurDemandee(Position_curseur);
                    }
                }
                if (e->kind_of_event() == sdl2::event::key_down)
                {
                    auto & key_ev = dynamic_cast<sdl2::event_keyboard&>(*e);
                    if ((key_ev.code() == SDLK_t) && (key_ev.state() == sdl2::event_mouse_button::pressed))
                    {
                        std::chrono::steady_clock::time_point temps_depart_radioprotection = std::chrono::steady_clock::now();
                        fenetre.cls({0, 0, 0, 0});
                        quitting = affichage_radioprotection(fenetre, centrale, jeu, dispatching, delta_tour_ms);
                        fenetre.cls({0, 0, 0, 0});
                        if (quitting == true)
                            return jeu.getMode();
                        std::chrono::steady_clock::time_point temps_arrivee_radioprotection = std::chrono::steady_clock::now();

                        // Opération permettant d'accorder un temps supplémentaire pour accéder à l'écran de radioprotection
                        // Petit inconvénient : le joueur doit optimiser son temps de jeu en restant in-extremis juste avant la fin du tour
                        // sur l'écran de radioprotection, on peut donc un peu "ralentir le temps" en spamant pour passer d'un écran à l'autre
                        temps_depart += temps_arrivee_radioprotection - temps_depart_radioprotection;

                        Tour.setNombre_de_tours(jeu.getNombre_de_tours());

                        fenetre << titre1 << titre3 << titre4 << titre5 << val_debit_eau_1 << val_pression_1 << temperature_circuit_primaire
                        << val_pression_enceinte << valeur_debit_eau_3 << val_temp_vap << val_debit_eau_2 << val_pression_2 << valeur_difference_de_temps
                        << Panneau << production << Tour;
                        if (jeu.getMode() != modes::Mode1 && jeu.getNombre_de_tours() > 0)
                        {
                            sdl2::texte ordre(dispatching.ordre(), fonte_soustitre, fenetre, {255, 255, 255, 255});
                            ordre.at(position_ordres[0], position_ordres[1]);
                            sdl2::rectangle rect_ordre(ordre.getPosition(), {static_cast<int>(ordre.dimension()[0]*10), ordre.dimension()[1]}, {0, 0, 0, 255}, true);
                            fenetre << rect_ordre << ordre << titre2;
                            if (jeu.getMode() == modes::Mode3)
                            {
                                sdl2::texte catastrophe(message_catastrophe, fonte_soustitre, fenetre, {234, 85, 251});
                                catastrophe.at(position_catastrophes[0], position_catastrophes[1]);
                                sdl2::rectangle rect_catastrophe(catastrophe.getPosition(), catastrophe.dimension()*std::array<int, 2ULL>{5, 1}, {0, 0, 0, 0}, true);
                                fenetre << titre_catastrophes << rect_catastrophe << catastrophe;
                            }
                        }
                        else
                        {
                            if (jeu.getMode() != modes::Mode1)
                            {
                                fenetre << titre2;
                                if (jeu.getMode() == modes::Mode3)
                                    fenetre << titre_catastrophes;
                            }
                        }

                    }
                    if ((key_ev.code() == SDLK_s) && (key_ev.state() == sdl2::event_keyboard::pressed))
                    {
                        jeu.fin_du_game = true;
                        return jeu.getMode();
                    }
                }
            }
            SDL_Delay(17);
            image.set({centrale.circuit_primaire.E_press() < tol, centrale.circuit_primaire.E_p1() < tol,
                       centrale.circuit_secondaire.E_vap() < tol, centrale.circuit_primaire.E_C1() < tol,
                       centrale.circuit_secondaire.E_C2() < tol, centrale.reacteur.E_cuve() < tol,
                       centrale.reacteur.E_barre() < tol, centrale.circuit_secondaire.E_p2() < tol,
                       centrale.circuit_secondaire.E_cd() < tol, centrale.E_enceinte() < tol, centrale.circuit_primaire.E_EC() < tol});
            image.update();
            fenetre << CacheImg << image;
            temps_arrivee = std::chrono::steady_clock::now();

            delta_temps = std::chrono::duration_cast<std::chrono::milliseconds>(temps_arrivee - temps_depart).count();
            Tour.setValeurActuelle(1 - delta_temps/static_cast<double>(delta_tour_ms));
            fenetre << Tour;
        }

        val_debit_eau_1.setValeurActuelle(centrale.circuit_primaire.D1());
        val_pression_enceinte.setValeurActuelle(centrale.P_enceinte());
        val_debit_eau_2.setValeurActuelle(centrale.circuit_secondaire.D2());
        valeur_debit_eau_3.setValeurActuelle(centrale.circuit_secondaire.D_cd());
        val_pression_2.setValeurActuelle(centrale.circuit_secondaire.P_vap());
        valeur_difference_de_temps.setValeurActuelle(centrale.circuit_secondaire.delta_ES());
        val_temp_vap.setValeurActuelle(centrale.circuit_secondaire.T_vap());
        temperature_circuit_primaire.setValeurActuelle(centrale.circuit_primaire.T1());
        val_pression_1.setValeurActuelle(centrale.circuit_primaire.P1());


        centrale.circuit_primaire.F_p1(val_pompe_1.getValeurDemandee());
        centrale.reacteur.Tx_Gr(valeur_etat_barre_graphite.getValeurDemandee());
        centrale.circuit_primaire.T_press(val_temperature_pressuriseur.getValeurDemandee());
        centrale.circuit_secondaire.F_p2(val_pompe_2.getValeurDemandee());
        centrale.reacteur.Tx_B(taux_de_bore.getValeurDemandee());
        centrale.circuit_secondaire.F_cd(valeur_pompe_3.getValeurDemandee());

        val_pompe_1.setValeurActuelleMin(plagedonnees::min_F_p1(centrale));
        val_temperature_pressuriseur.setValeurActuelleMin(plagedonnees::min_T_press(centrale));
        taux_de_bore.setValeurActuelleMin(plagedonnees::min_Tx_B(centrale));
        val_pompe_2.setValeurActuelleMin(plagedonnees::min_E_p2(centrale));
        valeur_pompe_3.setValeurActuelleMin(plagedonnees::min_F_cd(centrale));
        valeur_etat_barre_graphite.setValeurActuelleMin(plagedonnees::min_Tx_Gr(centrale));

        val_pompe_1.setValeurActuelleMax(plagedonnees::max_F_p1(centrale));
        val_temperature_pressuriseur.setValeurActuelleMax(plagedonnees::max_T_press(centrale));
        taux_de_bore.setValeurActuelleMax(plagedonnees::max_Tx_B(centrale));
        val_pompe_2.setValeurActuelleMax(plagedonnees::max_E_p2(centrale));
        valeur_pompe_3.setValeurActuelleMax(plagedonnees::max_F_cd(centrale));
        valeur_etat_barre_graphite.setValeurActuelleMax(plagedonnees::max_Tx_Gr(centrale));

        val_temperature_pressuriseur.setValeurActuelle(centrale.circuit_primaire.Tpress_eff());
        val_pompe_1.setValeurActuelle(centrale.circuit_primaire.F_p1());
        taux_de_bore.setValeurActuelle(centrale.reacteur.TB_eff());
        val_pompe_2.setValeurActuelle(centrale.circuit_secondaire.F_p2());
        valeur_pompe_3.setValeurActuelle(centrale.circuit_secondaire.F_cd());
        valeur_etat_barre_graphite.setValeurActuelle(centrale.reacteur.TGr_eff());

        Alertes = centrale.alertes();
        centrale.reparation();
        centrale.update();
        Tour.setNombre_de_tours(jeu.getNombre_de_tours());

        
        switch (jeu.getMode())
        {
            case (modes::Mode1):
                jeu.update(centrale.MW());
                jeu.fin_du_game = (centrale.etat() == Urgence::DEFCOM_4);
                break;
            case (modes::Mode2):
                jeu.update(centrale.MW(), dispatching.score());           // Commande pour le mode dispatching
                jeu.fin_du_game = dispatching.update(centrale.MW()) && (centrale.etat() == Urgence::DEFCOM_4);
                break;
            case (modes::Mode3):                                          // Commande pour le mode catastrophes
                jeu.update(centrale.MW(), dispatching.score());
                jeu.fin_du_game = dispatching.update(centrale.MW()) && (centrale.etat() == Urgence::DEFCOM_4);
                nouveau_message_catastrophe = centrale.catastrophe();
                break;
            default:
                jeu.setMode(modes::Void);
                return modes::Void;
                break;
        }
        if (jeu.getMode() != modes::Mode1)
        {
            sdl2::texte ordre(dispatching.ordre(), fonte_soustitre, fenetre, {255, 255, 255, 255});
            ordre.at(position_ordres[0], position_ordres[1]);
            sdl2::rectangle rect_ordre(ordre.getPosition(), {static_cast<int>(ordre.dimension()[0]*10), ordre.dimension()[1]}, {0, 0, 0, 255}, true);
            fenetre << rect_ordre << ordre;
            if (jeu.getMode() == affichage::modes::Mode3)
            {
                if (nouveau_message_catastrophe != "Jusqu'ici tout va bien... Enfin, pour le moment... " || jeu.getNombre_de_tours() == 1 || jeu.getNombre_de_tours() - compteur_cata >= 10) // On était fatigué, on a bourriné ici
                {    
                    message_catastrophe = nouveau_message_catastrophe;     
                    compteur_cata = jeu.getNombre_de_tours();
                }
                sdl2::texte catastrophe(message_catastrophe, fonte_soustitre, fenetre, {234, 85, 251});
                catastrophe.at(position_catastrophes[0], position_catastrophes[1]);
                sdl2::rectangle rect_catastrophe(catastrophe.getPosition(), catastrophe.dimension()*std::array<int, 2ULL>{5, 1}, {0, 0, 0, 0}, true);
                fenetre << rect_catastrophe << catastrophe;
            }
        }
        temps_depart = std::chrono::steady_clock::now();
    }
    jeu.setMode(modes::Void);
    return modes::Void;     // Ligne permettant d'éviter les warnings de notre cher compilateur
}


void fin_du_game(sdl2::window & fenetre, Centrale & centrale, Jeu & jeu)
{
    fenetre.cls({20, 20, 20, 20});
    auto [DIMx, DIMy] = fenetre.dimensions();

    affichage::point<int> position_etat_centrale{static_cast<int>(4*DIMx/10), static_cast<int>(DIMy/3.5)};
    affichage::point<int> position_production{static_cast<int>(4*DIMx/10), DIMy/3};
    affichage::point<int> position_mention_dispatching{static_cast<int>(4*DIMx/10), static_cast<int>(DIMy/2.5)};
    affichage::point<int> position_points_dispatching{static_cast<int>(DIMx/10), static_cast<int>(DIMy/2.5)};
    affichage::point<int> position_production_moyenne{static_cast<int>(DIMx/10), DIMy/3};
    affichage::point<int> position_contamination{static_cast<int>(4*DIMx/10), static_cast<int>(DIMy/1.4)};

    sdl2::color blanc{0xFF,0xFF,0xFF,0xFF};

    sdl2::font fonte_soustitre("./data/Lato-Bold.ttf", 20*DIMx/1000);
    sdl2::font fonte_titre("./data/Lato-Bold.ttf", 40*DIMx/1000);


    sdl2::texte titre1("FIN DE LA PARTIE", fonte_titre, fenetre, blanc);
    titre1.at(static_cast<int>(DIMx/2.7), static_cast<int>(DIMy/20));

    affichage::BarreEtat Etat_circuit_primaire_m("État circuit primaire", 0, 1, 1, {4*DIMx/10, static_cast<int>(DIMy/2)}, {static_cast<int>(1.5*DIMx/10), 2*DIMy/50});
    affichage::BarreEtat Etat_circuit_secondaire_m("État circuit secondaire", 0, 1, 1, {4*DIMx/10, static_cast<int>(DIMy/1.5)}, {static_cast<int>(1.5*DIMx/10), 2*DIMy/50});

    // ETAT CENTRALE
    affichage::BarreEtat Etat_centrale("État centrale", 0, 1, 1, {4*DIMx/10, static_cast<int>(DIMy/4.5)}, {static_cast<int>(1.5*DIMx/10), 2*DIMy/50});

    sdl2::texte texte1("la centrale est en bon état", fonte_soustitre, fenetre, blanc);
    texte1.at(position_etat_centrale[0], position_etat_centrale[1]);
    sdl2::texte texte2("La centrale assez endommagée", fonte_soustitre, fenetre, blanc);
    texte2.at(position_etat_centrale[0], position_etat_centrale[1]);
    sdl2::texte texte3("La centrale est presque détruite", fonte_soustitre, fenetre, blanc);
    texte3.at(position_etat_centrale[0], position_etat_centrale[1]);
    sdl2::texte texte4("La centrale est détruite", fonte_soustitre, fenetre, blanc);
    texte4.at(position_etat_centrale[0], position_etat_centrale[1]);

    // Radioactivités

    sdl2::texte texte_contamination_population1("Contamination radioactive légère de la population", fonte_soustitre, fenetre, blanc);
    texte_contamination_population1.at(position_contamination[0], position_contamination[1]);
    sdl2::texte texte_contamination_population2("Contamination radioactive importante de la population", fonte_soustitre, fenetre, blanc);
    texte_contamination_population2.at(position_contamination[0], position_contamination[1]);
    sdl2::texte texte_contamination_population3("Population gravement contaminée", fonte_soustitre, fenetre, blanc);
    texte_contamination_population3.at(position_contamination[0], position_contamination[1]);
    sdl2::texte texte_contamination_population4("Population non contaminée !", fonte_soustitre, fenetre, blanc);
    texte_contamination_population4.at(position_contamination[0], position_contamination[1]);

    double contamination = centrale.population.contamination();
    if (contamination <= 10)
    {
        fenetre << texte_contamination_population4;
    }
    if (contamination > 10 && contamination <= 600)
    {
        fenetre << texte_contamination_population1;
    }
    if (contamination > 600 && contamination <= 800)
    {
        fenetre << texte_contamination_population2;
    }
    if (contamination > 800)
    {
        fenetre << texte_contamination_population3;
    }


    // PRODUCTION
    sdl2::texte texte5("Hors norme", fonte_soustitre, fenetre, blanc);
    texte5.at(position_production[0], position_production[1]);
    sdl2::texte texte6("Excellente", fonte_soustitre, fenetre, blanc);
    texte6.at(position_production[0], position_production[1]);
    sdl2::texte texte7("Correcte", fonte_soustitre, fenetre, blanc);
    texte7.at(position_production[0], position_production[1]);
    sdl2::texte texte8("Peu rentable", fonte_soustitre, fenetre, blanc);
    texte8.at(position_production[0], position_production[1]);
    sdl2::texte texte8bis("Faible production", fonte_soustitre, fenetre, blanc);
    texte8bis.at(position_production[0], position_production[1]);

    // DISPATCHING
    sdl2::texte texte9("Nombre de points corrects", fonte_soustitre, fenetre, blanc);
    texte9.at(position_mention_dispatching[0], position_mention_dispatching[1]);
    sdl2::texte texte10("Nombre de points moyen", fonte_soustitre, fenetre, blanc);
    texte10.at(position_mention_dispatching[0], position_mention_dispatching[1]);
    sdl2::texte texte11("Nombre de points décevant", fonte_soustitre, fenetre, blanc);
    texte11.at(position_mention_dispatching[0], position_mention_dispatching[1]);

    // get nombre de points
    double points = jeu.getScore();

    double production_moyenne = jeu.getProduction_moyenne();

    // ETAT CENTRALE
    Etat_centrale.setValeurActuelle(centrale.E_centrale());
    Etat_circuit_primaire_m.setValeurActuelle(centrale.circuit_primaire.E_C1());
    Etat_circuit_secondaire_m.setValeurActuelle(centrale.circuit_secondaire.E_C2());
    switch (centrale.etat())
    {
        case (Urgence::DEFCOM_0):
            fenetre << texte1;
            break;
        case (Urgence::DEFCOM_1):
            fenetre << texte2;
            break;
        case (Urgence::DEFCOM_2):
            fenetre << texte3;
            break;
        default:
            fenetre << texte4;
            break;
    }

    // PRODUCTION MOYENNE
    if (production_moyenne > 1400)
        fenetre << texte5;
    if (1400 >= production_moyenne && production_moyenne > 1200)
        fenetre << texte6;
    if (1200 >= production_moyenne && production_moyenne > 900)
        fenetre << texte7;
    if (900 > production_moyenne && production_moyenne >= 300)
        fenetre << texte8;
    if (300 > production_moyenne)
        fenetre << texte8bis;




    sdl2::texte production_moyenne_texte("Production moyenne : " + std::to_string(static_cast<int>(production_moyenne)) + " MW", fonte_soustitre, fenetre, blanc);
    sdl2::texte points_dispatching("Points dispatching : " + std::to_string(static_cast<int>(points)), fonte_soustitre, fenetre, {255, 255, 255, 0});

    production_moyenne_texte.at(position_production_moyenne[0], position_production_moyenne[1]);
    points_dispatching.at(position_points_dispatching[0], position_points_dispatching[1]);


    //DISPATCHING

    if (jeu.getMode() != affichage::modes::Mode1)
    {
        fenetre << points_dispatching;
        if (points > 8)
            fenetre << texte9;
        if (8 >= points && points>5)
            fenetre << texte10;
        if (5 >= points)
            fenetre << texte11;
    }
    fenetre << Etat_circuit_primaire_m << Etat_centrale << Etat_circuit_secondaire_m << production_moyenne_texte << titre1 << sdl2::flush;

    bool quitting = false;
    sdl2::event_queue queue;

    while (!quitting)
    {
        auto events = queue.pull_events();

        for (const auto& e : events)
        {
            if (e->kind_of_event() == sdl2::event::quit)
            {
                quitting = true;
                break;
            }
        }
        SDL_Delay(34);  // Ici pas besoin de plus de 30 ips étant donné que l'affichage est statique
    }
}


int main(int argc, char* args[])
{
    Jeu jeu;
    unsigned delta_tour_ms = 1000;

    sdl2::init(argc, args);
    sdl2::window fenetre("Nuclear Alert", {1500, 800});

    // Affichage du menu
    jeu.setMode(affichage_modes(fenetre));
    fenetre.cls({0, 0, 0, 255});
    if (jeu.getMode() == affichage::modes::Void)
        return EXIT_SUCCESS;

    // Initialisation de la centrale pour commencer la partie
    Centrale centrale;

    affichage_salle_de_controle(fenetre, centrale, jeu, delta_tour_ms);

    // Affichage de la fin de partie si motivée
    if (jeu.getMode() != affichage::modes::Void)
        fin_du_game(fenetre, centrale, jeu);

    sdl2::finalize();

    return EXIT_SUCCESS;
}
