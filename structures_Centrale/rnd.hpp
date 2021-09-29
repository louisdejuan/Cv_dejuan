#ifndef _RND__HPP_
#define _RND__HPP_

#include <random>
#include <type_traits>



template <typename T>
concept integralType = std::is_integral<T>::value;


/* Générateur d'aléatoire. */
class Rnd
{
private:
    static std::random_device _rd;

public:
    static const double random(const double& x)
    {
        std::uniform_real_distribution<double> distrib(0, x);
        return distrib(_rd);
    }

    static const double random(const double& a, const double& b)
    {
        std::uniform_real_distribution<double> distrib(a, b);
        return distrib(_rd);
    }


    template <integralType T>
    static const T randomInteger(const T& x)
    {
        std::uniform_int_distribution<T> distrib(0, x);
        return distrib(_rd);
    }

    template <integralType T>
    static const T randomInteger(const T& a, const T& b)
    {
        std::uniform_int_distribution<T> distrib(a, b);
        return distrib(_rd);
    }
};



#endif
