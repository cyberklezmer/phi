#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions.hpp>
#include <assert.h>
#include <ctype.h>
#include <random>
#include <iostream>
#include <fstream>
#include <string>

#include <epp/defs.hpp>
#include <epp/io.hpp>
#include <epp/vector.hpp>

inline void logrealtime()
{
    time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );
    std::cout << (now->tm_hour) << ':'
         << (now->tm_min) << ':'
         << (now->tm_sec);
}



#define L( X) { logrealtime(); std::cout << " " << X << std::endl;}


using namespace epp;

template<typename I>
class distribution
{
public:
    virtual double pdf(I) const = 0;
    virtual double cdf(I) const = 0;
};

class zdistribution : public distribution<int>
{
double p;
    int l;
    vector <double> v;
    void setzeros()
    {
        for(unsigned int i = 0; i<v.size(); i++)
            v[i] = 0;
    }
    vector<double> temp;

public:
    static constexpr double llimit = -INT_MAX;
    static constexpr double ulimit = INT_MAX;

    zdistribution() : l(ulimit)
    {
    }
    zdistribution(int aleast, int agreatest) :
        l(aleast), v(agreatest-aleast+1)
    {
        setzeros();
    }

    int least() const { return l; }
    int greatest() const { return l+v.size()-1; }
    int range() const { return greatest()-least(); }
    void set(int aleast, vector<double>& avalues)
    {
        l = aleast;
        v = avalues;
    }
    void set(int aleast, int agreatest, bool azeros=true)
    {
        l = aleast;
        v.resize(agreatest-aleast+1);
        if(azeros)
            setzeros();
    }
    void setpdf(int i, double p)
    {
        if(i < least() || i > greatest())
            throw std::runtime_error("pdf out of range");
        v[i-least()] = p;
    }
    const vector<double>& getv() const
    {
        return v;
    }
    double pdf(int i) const
    {
        return i < least() ? 0 : (i>greatest() ? 0 : v[i-l]);
    }
    double quantile(double p)
    {
        double pi = 0;
        unsigned int i = 0;
        for(;i<v.size();i++)
        {
            pi += v[i];
            if(pi >= p)
                return least()+(signed)i;
        }
        assert(false);
        return 0;
    }
    virtual double cdf(int i) const
    {
        if(i > greatest())
            return 1.0;
        double p = 0.0;
        for(int j=least(); j <= i && j<=greatest(); j++)
            p += pdf(j);
        return p;
    }
    void truncate(int i)
    {
        if(i > greatest())
            throw std::runtime_error("Truncation at point with probability one");
        if(i > least())
        {
            double p=0;
            int j;
            for(j=least(); j<i && j<=greatest(); j++)
                p+=pdf(j);
            if(p==1.0)
                throw std::runtime_error("Truncation at x with cdf(x)==1.");
            for( ;j<=greatest(); j++)
                v[j-i] = pdf(j) / (1.0-p);
            set(i,greatest(),false);
        }
    }

    void scale(double aa, double* err = 0)
    {
        double e = 0;
        if(aa==0)
        {
            set(0,0);
            v[0] = 1;
        }
        else
        {
            int nl;
            double ng;
            double aabs = fabs(aa);
            if(aabs >= 1)
            {
                nl = round(least() * aabs);
                ng = round(greatest() * aabs);
                temp.resize(ng-nl+1);
                for(unsigned int i=0; i<temp.size(); i++)
                    temp[i] = 0;
                for(int i=least(); i<=greatest(); i++)
                    temp[round(i * aabs)-nl] = pdf(i);
            }
            else
            {
                nl = floor(least() * aabs);
                ng = ceil(greatest() * aabs);
                assert(ng-nl+1 > 0);
                temp.resize(ng-nl+1);
                int j = least();
                for(int i=nl; i<=ng; i++)
                {
                    double p1 = 0;
                    while(((j+0.5)*aabs <= i) && j <= greatest())
                        p1+=pdf(j++);
                    double p2 = 0;
                    while(((j+0.5)*aabs <= i+0.5) && j <= greatest())
                        p2+=pdf(j++);
                    temp[i-nl] = p1+p2;
                    double em=std::max(p1,p2);
                    if(em > e)
                        e = em;
                }
                assert(j<=greatest()+1);
                assert(j>=greatest()+1);
            }
            if(aa > 0)
                set(nl,temp);
            else
            {
                set(-ng,-nl);
                for(unsigned int j=0; j<temp.size(); j++)
                    v[temp.size()-j-1]=temp[j];
            }

        }
        if(err)
            *err = e;
    }

    void shift(double y)
    {
        l+= (int) (y+0.5);
    }

    void approx(double maxkolmogorov)
    {
        assert(maxkolmogorov < 1);

        int i=least();
        int j=greatest();
        double left=0;
        double right=0;
        for(;;)
        {
            if(pdf(i) < pdf(j))
            {
                if(left+right+pdf(i) > maxkolmogorov)
                    break;
                else
                    left += pdf(i++);
            }
            else
            {
                if(left+right+pdf(j) > maxkolmogorov)
                    break;
                else
                    right += pdf(j--);
            }
        }
        v[i-least()]+= left;
        v[j-least()]+=right;
        int k=least();
        int l=i;
        for(; l<=j; k++,l++)
            v[k-least()]=v[l-least()];
        set(i,j,false);
    }

    double sump() const
    {
        return cdf(greatest());
    }

    double mean() const
    {
        double sum = 0;
        for(unsigned int i=0; i<v.size(); i++)
            sum += v[i]*(double)i;
        return least() + sum;
    }

    double moment(int p) const
    {
        double sum = 0;
        for(unsigned int i=0; i<v.size(); i++)
            sum += v[i]*pow(least()+(signed)i,p);
        return sum;
    }

    double var() const
    {
        double m = mean();
        return moment(2) - m*m;
    }

    zdistribution& operator +=(zdistribution& s)
    {
        int nl = s.least() + least();
        int ng = s.greatest() + greatest();
        temp.resize(ng-nl+1);
        for(unsigned int i=0; i<temp.size(); i++)
            temp[i] = 0;
        double *p=s.v.data();
        for(unsigned int i=0; i<s.v.size(); i++,p++)
        {
            double *q = v.data();
            double *t = temp.data()+i;
            for(unsigned int j=0; j<v.size(); j++,q++,t++)
                *t+=(*p)*(*q);
        }
        set(nl,temp);
        return *this;
    }
    vector<double> histogram(const vector<double>& frontiers, double scale=1)
    {
        vector<double> res = zero_vector(frontiers.size()+1);
        int x=least();
        double p=0;
        unsigned i=0;
        for(; i<frontiers.size() && x <= greatest(); i++)
        {
            while(x <= frontiers[i]*scale)
                res[i]+=pdf(x++);
            p+=res[i];
        }
        res[i]=1-p;
        return res;
    }
};

inline std::ostream& operator<<(std::ostream& str,const zdistribution& d)
{
    str << "[" << d.least() << ".." << d.greatest() << "]" << std::endl;
    str << d.getv();
    str << std::endl;
    return str;
};







//void Phistep(z)


class mftrans
{
    unsigned T;
    unsigned m;
    vector<double> Q;
    vector<double> Y;
    vector<double> G;
    vector<double> I;
    vector<double> nc;
    vector<double> upsilons;
    vector<double> C;
    vector<double> Qerr;
    vector<double> Yerr;
    bool singleupsilon;
    bool ystarinput;


    double sigma;
    double sigma1;
    double phi;
    double rho;
    double psi;

    double dprec;

    /// set by \p transform
    double scale;
    vector<vector<double>> ws;
    vector<zdistribution> W;

    std::ofstream dlog;
    vector<double> dfronts;


    double getnorm()
    {
        static std::mt19937 gen;

        std::normal_distribution<> ndist(0,1);
        return ndist(gen);
    }


    double normalcdf(double a)
    {
        static boost::math::normal_distribution<> norm;
        return boost::math::cdf(norm,a);
    }

    double normalq(double p)
    {
        static boost::math::normal_distribution<> norm;
        return boost::math::quantile(norm, p);
    }

    double minnormdisterror(double sigma)
    {
        return normalcdf(0.5/sigma)-0.5;
    }

    double minnormsdistscale(double err)
    {
        return 0.5/normalq(err + 0.5);
    }

    double ystarcor(int s, int t)
    {
        return log(C[t]) - log(C[s])
              - log((t+1-s)*b(s));
    }


    int yrawcor(int s, int t, double scale)
    {
        return ystarinput ? 0 : round(scale * ystarcor(s,t));
    }

    double ystarraw(double yraw, int s, int t, double scale)
    {
        return yraw + yrawcor(s,t, scale);
    }

    void setq(vector<double>& qs,vector<zdistribution>& D,
                int firstnz, unsigned t, int yraw, double scale)
    {
        for(unsigned j=firstnz; j<=t; j++)
            qs[j] = D[j].cdf(-ystarraw(yraw,j,t,scale));
    }



    void discretizenormal(zdistribution& d, double sigma, double& err)
    {
        double minerr = minnormdisterror(sigma);
        if(minerr > err*1.00000001)
        {
            L("warning: minerr=" << minerr << "> err=" << err)
            err = minerr;
        }
        int l = ceil(-normalq(err)*sigma-0.5);

    //    L("discr: err=" << err << " n=" << l*2+1 << " chosen");

        double lowp = 0;
        d.set(-l,l);
        for(int i=-l; i<=0; i++)
        {
            double highp = normalcdf((i+0.5)/sigma);
            d.setpdf(i,highp-lowp);
            d.setpdf(-i,highp-lowp);
            lowp=highp;
        }
    }

    double v(int s)
    {
        return 1.0 / (1.0 + upsilons[s]);
    }
    double b(int s)
    {
        return upsilons[s] == 0
              ? 1.0 / (double) m
              : upsilons[s] / (1.0 - pow(v(s),m));
    }

    double ht(int s, int t)
    {
        return upsilons[s] == 0
             ? 1.0
             : b(s) * v(s) * (1-pow(v(s),m-t)) / (1.0 - v(s));
    }

    double rhot(double t)
    {
        return rho * sqrt((1.0-pow(psi,t+1)) / (1.0-psi));
    }

    double f(int t, double x)
    {
        double rt = rhot(t);
        double res = normalcdf(-x / rt)
           - exp( x+rt*rt / 2.0) * normalcdf( (-x-rt*rt)/rt );

        if(res > 1)
            throw std::logic_error("f > 1");
        return res;
    }


    double ItoG(const vector<double>& w, const vector<double>& qs,
                int firstnz, unsigned t,
                double i)
    {
        double rnum=0;
        double rden=0;
        for(unsigned j=firstnz; j<=t; j++)
        {
            double h = ht(j,t);
            assert(h > 0);
            double istart = j==0 ? 0.0 : I[j-1];
            rnum += w[j]*qs[j]*f(t,i-istart-log(h));
            rden += w[j];
        }
        return rnum / rden;
    }

public:

    enum etrsfway {FtoR, RtoF};

    void transform(etrsfway aw, bool aIs)
    {
        scale = minnormsdistscale(dprec) / std::min(sigma1,sigma) ;

        vector<int> yraw(T);
        vector<zdistribution> D(T);

        if(aw==FtoR)
            for(unsigned int t=0; t<T; t++)
                yraw[t] = round(scale * Y[t]);

        zdistribution u;
        discretizenormal(u,sigma*scale,dprec);

        vector<double> errs = zero_vector(T);
        vector<double> w = zero_vector(T);
        vector<double> qs = zero_vector(T);

        dlog << "Fronts" << dfronts << std::endl;

        for(unsigned int t=0;;t++)
        {
            if(nc[t]>0)
            {
                discretizenormal(D[t],sigma1*scale,dprec); // initial
                errs[t] = dprec;
            }
            else
            {
                D[t].set(0,0);
                D[t].setpdf(0,1);
                errs[t] = 0;
            }
            w[t] = nc[t];
            assert(m != 1 || nc[t] == 1);


            double q; // weight to remove
            int firstnz;
            if((signed) t-(signed)m >= 0)
            {
                q = w[t-m]; // those who end
                w[t-m] = 0;
                firstnz = t-m+1;
            }
            else
            {
                q = 0;
                firstnz = 0;
            }

            if(q==1.0)
            {
                if(nc[t] != 1.0)
                    throw std::logic_error(
                     "Portfolio has emptied  while newcomers rate <1");
                else
                   w[t] = 1.0; // other weights have to be zero
            }
            else
            {
                double ws = 0;
                for(unsigned int j=firstnz; j<t; j++)
                    ws += w[j];
                if(nc[1] == 1.0 && ws > 0)
                    throw std::logic_error(
                     "Portfolio not empty while newcomers rate =1");
                for(unsigned int j=firstnz; j<t; j++)
                    w[j] *= (1.0-nc[t])/(1.0-q);
            }

            if(aw==RtoF)
            {
                if(Q[t]<0 || Q[t]>1)
                    throw std::runtime_error("Q out of [0,1]");
                else if(Q[t] == 0.0)
                    yraw[t] = INT_MAX;
                else if(Q[t] == 1.0)
                    yraw[t] = INT_MIN;
                else
                {
                    vector<int> cors(T);

                    int maxyraw = INT_MIN;
                    int minyraw = INT_MAX;
                    double deltat = 0.0;

                    for(unsigned int j=firstnz; j<=t; j++)
                        if(w[j])
                        {
                            int cor = yrawcor(j,t,scale);
                            cors[j] = cor;
                            int maxcandidate = -D[j].least()-cor;
                            if(maxcandidate > maxyraw)
                                maxyraw = maxcandidate;
                            int mincandidate = -D[j].greatest()-cor;
                            if(mincandidate < minyraw)
                                minyraw = mincandidate;
                            deltat += w[j] * errs[j];
                        }
                    Qerr[t] = deltat;
                    double pi = 0;
                    int uyr = INT_MAX;
                    int yr;
                    int lyr = INT_MIN;
                    bool uset = false;
                    bool yset = false;

                    int y = maxyraw;
                    for(;y >= minyraw;y--)
                    {
                        for(unsigned int j=firstnz; j<=t; j++)
                            if(w[j])
                                pi += w[j]*D[j].pdf(-(y+cors[j]));
                        if(pi>=Q[t]-deltat)
                            if(!uset)
                            {
                                uyr = y;
                                uset = true;
                            }
                        if(pi>=Q[t])
                            if(!yset)
                            {
                                yr = y;
                                yset = true;
                            }
                        if(pi>=Q[t]+deltat)
                        {
                            lyr = y;
                            break;
                        }
                    }
                    if(pi < Q[t])
                        throw std::logic_error("pi < Q(t)");
                    yraw[t] = yr;
                    Yerr[t] = ((double) uyr - (double) lyr) / scale;
                }
                std::cout << "Y[" << t << "]=" << yraw[t] / scale;
//                L("G[" << t << "]=" << G[t]);
                if(aIs)
                {
                    if(G[t] >= Q[t])
                        throw std::runtime_error("Q < G");
                    if(G[t]<0 || G[t]>1)
                        throw std::runtime_error("G out of [0,1]");
                    else if(G[t] == 0.0)
                        I[t] = HUGE_VAL;
                    else if(G[t] == 1.0)
                        I[t] = -HUGE_VAL;
                    else
                    {
                        setq(qs,D,firstnz,t,yraw[t],scale);
//                        L("w=" << w)
//                        L("q=" << qs)
                        double i = I[t];
                        double g = ItoG(w,qs,firstnz,t, i)-G[t];
                        double d = g > 0 ? 1.0 : -1.0;
                        double oldd = 0.0;
                        double hi,lo;
                        for(;;)
                        {
                            g = ItoG(w,qs,firstnz,t,i+d)-G[t];
//                            L("i+d=" << i+d << " g=" << g << " f=" << f(t,i+d))
                            if(d > 0 && g < 0)
                            {
                                lo = i+oldd;
                                hi = i+d;
                                break;
                            }
                            if(d < 0 && g > 0)
                            {
                                lo= i+d;
                                hi= i+oldd;
                                break;
                            }
                            if(fabs(d) > rho * 100)
                                throw std::logic_error("Cannot find bound of interval whin inverting G");
                            oldd=d;
                            d *= 2.0;
                        }
                        for(;;)
                        {
                            double mi=(hi+lo) / 2.0;
                            g = ItoG(w,qs,firstnz,t, mi)-G[t];
                            if(g > 0)
                                lo=mi;
                            else
                                hi=mi;
                            if((hi-lo) < dprec)
                                break;
                        }
                        I[t] = (hi+lo) / 2.0;
                    }
                    std::cout << ", I[" << t << "]=" << I[t];
                }
                std::cout << std::endl;
            }
            else
            {
                Q[t] = 0.0;
                Qerr[t] = 0.0;
                for(unsigned int j=firstnz; j<=t; j++)
                    if(w[j])
                    {
                        Q[t] += w[j]*D[j].cdf(-ystarraw(yraw[t],j,t,scale));
                        Qerr[t] += w[j]*errs[j];
                    }
                std::cout << "Q[" << t << "]=" << Q[t];

                if(aIs)
                {
                    setq(qs,D,firstnz,t,yraw[t],scale);

                    G[t] = ItoG(w,qs,firstnz,t, I[t]);
                    std::cout << ", G[" << t << "]=" << G[t];
                }

                std::cout << std::endl;
            }

            // now set the output : C and w
            int cl = INT_MAX;
            int ch = -INT_MAX;
            for(unsigned int j=firstnz; j<=t; j++)
                if(w[j])
                {
                    int l = D[j].least();
                    int h = D[j].greatest();
                    if(l < cl)
                        cl = l;
                    if(h > ch)
                        ch = h;
                }
            int yshift = yraw[t] - (ystarinput ? yrawcor(0,t,scale) : 0);

            W[t].set(cl+yshift,ch+yshift,true);
            for(int k=cl; k<=ch; k++)
            {
                double p = 0;
                for(unsigned j=firstnz; j<=t; j++)
                    p += w[j]*D[j].pdf(k);
                W[t].setpdf(k+yshift,p);
            }
            dlog << "D[0]" << D[0].histogram(dfronts,scale) << std::endl;
            dlog << "W[t]" << W[t].histogram(dfronts,scale) << std::endl;
            ws[t] = w;

            // perform the transformations of D's
            for(unsigned int j=firstnz; j<=t; j++)
                if(w[j])
                {
                    int truncat = -ystarraw(yraw[t],j,t,scale);
                    D[j].truncate(truncat);
                    errs[j] = 2 * errs[j] / (1-D[j].cdf(truncat));

                    double scaleerr;
                    D[j].scale(phi,&scaleerr);

                    errs[j] += scaleerr;

                    D[j] += u;

                    errs[j] += dprec;
/*std::cout << "range(D[" << j << "])/2="
                  << D[j].greatest() - D[j].least()
                  << " w[j]=" << w[j]
                  << std::endl;*/
                }

            if(t+1 == T)
                break;
            for(unsigned int j=firstnz; j<=t; j++)
                if(w[j])
                {
                    D[j].approx(dprec);
                    errs[j] += dprec;
                }
        }

        if(aw==RtoF)
        {
            for(unsigned int t=0; t<yraw.size(); t++)
                Y[t] = yraw[t] / scale;
        }
        else
            Yerr = zero_vector(T);
    }

    void sim(vector<double> *aQ, vector<double> *aG = 0)
    {
        aQ->resize(T);
        if(aG)
            aG->resize(T);
        const int kg = 500000;
        vector<vector<double>> Z(T);
        vector<vector<double>> E(T);
        for(unsigned t=0; t<T; t++)
        {
            Z[t] = zero_vector(kg);
            if(aG)
                E[t] = zero_vector(kg);
        }

        for(unsigned int t=0; t<T; t++)
        {
            int firstnz = t-m;
            if(firstnz < 0)
                firstnz = 0;
            int N = 0;
            for(unsigned int j=firstnz; j<t; j++)
            {
                for(int k=0; k<kg; k++)
                {
                    double& wlth = Z[j][k];
                    if(!isnan(wlth))
                    {
                        if(aG)
                        {
                            double& price = E[j][k];
                            price = psi * price + getnorm()*rho;
                        }
                        wlth = phi * wlth + getnorm() * sigma;
                        N++;
                    }
                }
            }
            int nnew = t ==0 ? kg: round(N * nc[t]);
            assert(nnew<=kg);
            int k;
            for(k=0; k<nnew; k++)
            {
                Z[t][k] = getnorm()*sigma1;
                if(aG)
                    E[t][k] = getnorm()*rho;
                N++;
            }
            for(;k<kg; k++)
                Z[t][k]=NAN;
            int n = 0;
            double losses = 0;
            double portfolio = 0;

            for(unsigned j=firstnz; j<=t; j++)
            {
                for(int k=0; k<kg; k++)
                {
                    double&  wlth = Z[j][k];
                    if(!isnan(wlth))
                    {
                        if(wlth + Y[t]+(ystarinput ? 0 :  ystarcor(j,t)) < 0)
                        {
                            wlth = NAN;

                            if(aG)
                            {
                                double i = j==0 ? I[t] : I[t]-I[j-1];
                                double P = exp(i+E[j][k]);
                                losses += std::max(0.0,ht(j,t)-P);
                            }
                            n++;
                        }
                        portfolio += ht(j,t);
                    }
                }
            }
            (*aQ)[t] = (double) n / (double) N;
            if(aG)
                (*aG)[t] = losses / portfolio;
        }
    }


    mftrans(unsigned aT, unsigned am) : T(aT), m(am),
        Q(T), Y(T), G(T), I(T), nc(T), upsilons(T), C(T),
        Qerr(T), Yerr(T),
        ystarinput(false),
        sigma(1), sigma1(1), phi(0), rho(1), psi(0),
        dprec(0.0001), scale(NAN), ws(T), W(T),
        dlog("dlog.csv"), dfronts(51)
    {
        if(T == 0)
            throw std::runtime_error("Zero T.");
        if(m == 0)
            throw std::runtime_error("Zero n.");

        for(unsigned i=0; i<T; i++)
        {
            Q[i] = 0;
            G[i] = 0;
            Y[i] = 0;
            I[i] = 0;
            upsilons[i] = 0;
            nc[i] = 0;
            C[i] = 1;
            Yerr[i] = 0;
            Qerr[i] = 0;
        }
        nc[0] = 1;
        singleupsilon = true;

        double h = 0.1;
        double dfh = (double) dfronts.size() / 2.0 * h;
        for(unsigned z=0; z<dfronts.size(); z++)
            dfronts[z] = -dfh + z*h;
    }
    void setsigma(double asigma)
    {
        sigma = asigma;
    }
    void setsigma1(double asigma1)
    {
        sigma1 = asigma1;
    }
    void setphi(double aphi)
    {
        phi = aphi;
    }
    void setrho(double arho)
    {
        rho = arho;
    }
    void setpsi(double apsi)
    {
        psi = apsi;
    }

    double getsigma()
    {
        return sigma;
    }

    double getsigma1()
    {
        return sigma1;
    }


    double getphi()
    {
        return phi;
    }
    double getrho()
    {
        return rho;
    }
    double getpsi()
    {
        return psi;
    }

    unsigned getm()
    {
        return m;
    }
    unsigned getT()
    {
        return T;
    }


    void setY(const vector<double>& aY)
    {
        assert(aY.size()==T);
        Y = aY;
    }
    void setI(const vector<double>& aI)
    {
        assert(aI.size()==T);
        I = aI;
    }
    void setQ(const vector<double>& aQ)
    {
        assert(aQ.size()==T);
        Q = aQ;
    }
    void setG(const vector<double>& aG)
    {
        assert(aG.size()==T);
        G = aG;
    }
    void setnc(const vector<double>& anc)
    {
        assert(anc.size()==T);
        if(anc[0] == 0)
            throw std::logic_error("Zero newcomers at time zero not allowed");
        for(unsigned i=0; i<T; i++)
        {
            if(anc[i] < 0 || anc[i] >1)
                throw std::logic_error("Newcomers rate have to be in [0,1]");
            if(i>0 && m>1 && anc[i]==1)
                throw std::logic_error("Newcomers cannot be 1 if m>1");
        }
        nc = anc;
    }

    void setC(const vector<double>& aC)
    {
        assert(aC.size()==T);
        for(unsigned i=0; i<T; i++)
            if(aC[i] <= 0)
                throw std::logic_error("Inflation rate C has to be positive");
        C = aC;
    }

    void setupsilons(const vector<double>& au)
    {
        assert(au.size()==T);
        upsilons = au;
        double u = upsilons[0];
        singleupsilon = true;
        for(unsigned i=1; i<T; i++)
            if(upsilons[i]!=u)
            {
                singleupsilon = false;
                break;
            }
    }
    void setsingleupsilon(double au)
    {
        for(unsigned i=0; i<T; i++)
            upsilons[i] = au;
        singleupsilon = true;
    }
    bool issingleupsilon()
    {
        return singleupsilon;
    }
    void setystarinput(bool ayi)
    {
        if(ayi)
        {
            bool right = nc[0]==1;
            for(unsigned i=1; i<T; i++)
                if(nc[i]!=0)
                {
                    right = true;
                    break;
                }
            if(!right)
                throw std::logic_error("Ystarinput works only with signle portfolio.");
        }
        ystarinput = ayi;
    }
    bool isystarinput(bool ayi)
    {
        return ystarinput;
    }


    double getsingleupsilon()
    {
        if(!singleupsilon)
            throw std::logic_error("Upsilon not unique.");
        return upsilons[0];
    }

    vector<double> getY()
    {
        if(ystarinput)
        {
            vector<double> res(T);
            for(unsigned i=0; i<T; i++)
                res[i] = Y[i] - ystarcor(0,i);
            return res;
        }
        else
            return Y;
    }
    vector<double> getYstar()
    {
        if(ystarinput)
            return Y;
        else
        {
            vector<double> res(T);
            for(unsigned i=0; i<T; i++)
                res[i] = Y[i] + ystarcor(0,i);
            return res;
        }
    }

    const vector<double>& getI()
    {
        return I;
    }
    const vector<double>& getQ()
    {
        return Q;
    }
    const vector<double>& getG()
    {
        return G;
    }

    const vector<double>& getnc()
    {
        return nc;
    }
    const vector<double>& getYerr()
    {
        return Yerr;
    }
    const vector<double>& getQerr()
    {
        return Qerr;
    }



    const zdistribution& getlogW(unsigned int t)
    {
        return W[t];
    }
    double getlogWmean(unsigned t)
    {
        return W[t].mean() / scale;
    }

    double getlogWsigma(unsigned t)
    {
        return sqrt(W[t].var()) / scale;
    }

    void setfluentnc()
    {
        if(ystarinput)
            throw std::logic_error("Fluent nc and ystarinput cannot be combined.");
        unsigned i=0;
        for(; i<m && i<T; i++)
            nc[i] = 1.0 / (double) (i+1);
        for(; i<T; i++)
            nc[i] = 1.0 / (double) m;
    }

    void setstationarysigma1()
    {
        sigma1 =  sigma / sqrt(1-phi*phi);
    }

    void setprec(double aprec)
    {
        dprec = aprec;
    }

    double getprec()
    {
        return dprec;
    }


};

#define khelp "phi - ver 1.0" << endl << endl << \
"Program arguments:" << endl <<\
"i=INPUT   - input csv file (mandatory)" << endl <<\
"o=OUPTUT  - output csv file (mandatory)"<< endl <<\
"PAR=VALUE - PAR=sigma1/sigma/phi/rho/psi, VALUE=real number"<< endl <<\
"m=VALUE   - set duration of martgage to VALUE (if not specified, length of series is used)"<< endl <<\
"U=VALUE   - sets a single interest rate for all periods"<< endl <<\
"inv       - inverse transformation is computed" << endl <<\
"singleg   - only single portfolio" << endl <<\
"autonc    - set newcomers ratest to 1,1/2,1/3,...,1/m,1/m" << endl <<\
"precf=VAL - increase precision by VAL"  << endl <<\
"invc      - checks the computation by backward inversion"  << endl <<\
"simc=N    - checks the computation by simulation of size N"  << endl <<\
endl << "Input file columnts:" << endl <<\
"Y     - factor Y (mandatory without `inv', used as guess values with `inv')" << endl <<\
"I     - factor I (used as guess values with `inv')"  << endl <<\
"Q     - rate Q (mandatory with `inv')"<< endl <<\
"G     - factor G" << endl <<\
"N     - newcomers rates (if not specified, either `autonc' or `sinhleg' must be used)" << endl <<\
"C     - inflation/cpi (if not specified, 1 is used)" << endl <<\
"U     - interest rate (if not specified and `U' is not used, it is set to zero)" << endl <<\
"Ystar - Y^\\star, alternative to Y (may be used only with `singleg')" << endl <<\
endl << "Output file columnts:" << endl <<\
"Y/I   - results (with `inv'), copy of input (without `inv')" << endl <<\
"Q/G   - results (without `inv'), copy of input (with `inv')" << endl <<\
"Qe    - delta_t (bounds of errors of Q approximation)" << endl <<\
"Ye    - bounds of errors of Y approximation" << endl <<\
"N     - newcomers rates" << endl <<\
"M,S   - mean and stdev of wealth distribution" << endl <<\
endl << \
"YI,II,QI,GI - results of inversion check"<< endl <<\
"QS,GS,Qe,Ge - results and stdevs of simulation"<< endl


#define E(X) { stringstream s; s << "Error: " << X << endl; throw logic_error(s.str()); }


int main(int argc, char ** argv)
{
    using namespace std;
    if(argc==2 && argv[1][0] == '?')
        cout<<khelp << endl;
    else try
    {

        string input;
        string output;
        bool inversion = false;
        bool simcheck = false;
        int simn = 0;
        bool invcheck = false;
        bool ystarinput = false;
        bool autonc = false;
        bool singleg = false;
        unsigned m=0;
        double sigma = NAN;
        double sigma1 = NAN;
        double phi = NAN;
        double rho = NAN;
        double psi = NAN;
        double precfactor = NAN;
        double singleupsilon = NAN;
        for(int i=1; i<argc; i++)
        {
            stringstream s(argv[i]);

            string a;
            getline(s,a,'=');

            if(a == "i")
                s >> input;
            else if(a == "o")
                s >> output;
            else if(a == "inv")
                inversion = true;
            else if(a == "simc")
            {
                simcheck = true;
                if(s.eof())
                    simn = 10;
                else
                    s >> simn;
            }
            else if(a == "invc")
                invcheck = true;
            else if(a == "autonc")
                autonc = true;
            else if(a == "singleg")
                singleg = true;
            else if(a == "m")
                s >> m;
            else if(a == "U")
                s >> singleupsilon;
            else if(a == "sigma")
                s >> sigma;
            else if(a == "sigma1")
                s >> sigma1;
            else if(a == "phi")
                s >> phi;
            else if(a == "rho")
                s >> rho;
            else if(a == "psi")
                s >> psi;
            else if(a == "precf")
                s >> precfactor;
            else
                E("Unknown argument " << a)
        }
        if(input == "")
            E("Input file not selected")
        if(output == "")
            E("Output file not selected")
        if(simcheck && inversion)
            E("Simulation check not implemented for inversion")

        icsvstream inp(input);
        if(!inp)
            E("Cannot open input "<<input)

        ofstream out(output);
        if(!inp)
            E("Cannot open onput "<< output)

        csvrow ih;
        inp >> ih;
        int Ii = -1;
        int Yi = -1;
        int Ystari = -1;
        int Qi = -1;
        int Gi = -1;
        int Ui = -1;
        int Ci = -1;
        int Ni = -1;

        for(unsigned i=0; i<ih.size(); i++)
        {
            string lab = ih[i];
            if(lab == "I")
                Ii = i;
            else if(lab == "Y")
                Yi = i;
            else if(lab == "Ystar")
                Ystari = i;
            else if(lab == "G")
                Gi = i;
            else if(lab == "Q")
                Qi = i;
            else if(lab == "U")
                Ui = i;
            else if(lab == "C")
                Ci = i;
            else if(lab == "N")
                Ni = i;
            else
                E("Unknown label " << lab)
        }
        if(Yi>=0 && Ystari>=0)
            E("Y and Ystar cannot be supplied simultaneously")
        if(Ui>=0 && singleupsilon)
            E("Conflicting input of U")

        unsigned T = 0;
        epp::vector<double> I;
        epp::vector<double> Y;
        epp::vector<double> Q;
        epp::vector<double> G;
        epp::vector<double> U;
        epp::vector<double> C;
        epp::vector<double> N;

        for(;!inp.eof();)
        {
            csvrow ir;
            inp >> ir;
            if(ir.size() == 0)
                break;
            T++;
            if(ir.size() != ih.size())
                E("Bad format of input, line=" << T+1
                   << " header size=" << ih.size() << " row size=" << ir.size())
            for(int i=0; i<(signed)ih.size(); i++)
            {
                stringstream s(ir[i]);

                double v;
                s >> v;

                if(i == Ii)
                    I.push_back(v);
                else if(i == Yi)
                    Y.push_back(v);
                else if(i == Ystari)
                {
                    Y.push_back(v);
                    ystarinput = true;
                }
                else if(i == Qi)
                    Q.push_back(v);
                else if(i == Gi)
                    G.push_back(v);
                else if(i == Ui)
                    U.push_back(v);
                else if(i == Ci)
                    C.push_back(v);
                else if(i == Ni)
                    N.push_back(v);

            }
        }

        bool twodim = true;
        if(!inversion)
        {
            if(Yi < 0)
                E("Y not found")
            if(Ii < 0)
                twodim = false;
        }
        else
        {
            if(Qi < 0)
                E("Q not found")
            if(Gi < 0)
                twodim = false;
        }

        if(m <= 0)
            m = T;

        mftrans tr(T,m);

        if(Yi >= 0)
            tr.setY(Y);
        if(Ii >= 0)
            tr.setI(I);
        if(Qi >= 0)
            tr.setQ(Q);
        if(Gi >= 0)
            tr.setG(G);
        if(Ui >= 0)
            tr.setupsilons(U);
        if(Ci >= 0)
            tr.setC(C);

        int ncways = 0;
        if(autonc)
        {
            tr.setfluentnc();
            ncways++;
        }
        if(Ni >= 0)
        {
            tr.setnc(N);
            ncways++;
        }
        if(singleg)
            ncways++; // default for tr.
        if(ncways == 0)
            E("Not clear abou newcommers. Set either autonc, singleg or provice columnt N")
        else if(ncways > 1)
            E("Amgiguous input of newcomers");

        bool multig = false;
        for(unsigned i=1;i<T;i++)
            if(tr.getnc()[i] > 0)
            {
                multig = true;
                break;
            }

        if(ystarinput && multig)
            E("Ystar input cannot be combined with multiple generations")

        if(Ni>=0 && !multig)
            cout << "With single generation, N will not be taken into actcout."
                 << endl;

        if(Ui>=0 && !multig)
            cout << "With single generation, only the first value of U will be taken into actcout."
                 << endl;

        if(!std::isnan(singleupsilon))
            tr.setsingleupsilon(singleupsilon);
        if(ystarinput)
            tr.setystarinput(true);

        if(!std::isnan(sigma))
            tr.setsigma(sigma);
        if(!std::isnan(sigma1))
            tr.setsigma1(sigma1);
        if(!std::isnan(phi))
            tr.setphi(phi);
        if(!std::isnan(rho))
            tr.setrho(rho);
        if(!std::isnan(psi))
            tr.setpsi(psi);
        if(!std::isnan(precfactor))
        {
            if(precfactor == 0)
                E("precfactor cannot be zero.")
            tr.setprec(tr.getprec()/precfactor);
        }
        cout << "Processing" << endl;
    /*    cout << "Y=" << tr.getY() << endl;
        cout << "I=" << tr.getI() << endl;
        cout << "Q=" << tr.getQ() << endl;
        cout << "G=" << tr.getG() << endl;
        cout << endl;*/
        cout << "sigma=" << tr.getsigma() << ", ";
        cout << "sigma1=" << tr.getsigma1() << ", ";
        cout << "phi=" << tr.getphi() << endl;
        cout << "rho=" << tr.getrho() << ", ";
        cout << "psi=" << tr.getpsi() << ", ";
        cout << "upsilon=";
        if(tr.issingleupsilon())
            cout << tr.getsingleupsilon();
        else
            cout << "time varying";

        tr.setystarinput(ystarinput);

        cout << endl;
        cout << "T=" << tr.getT()  << ", ";
        cout << "m=" << tr.getm()  << endl;
        cout << "inversion=" << inversion << ", ";
        cout << "ystarinput=" << ystarinput << ", ";
        cout << "multig=" << multig << ", ";
        cout << "2D=" << twodim << endl;

        time_t st = time(0);
        tr.transform(inversion ? tr.RtoF : tr.FtoR, twodim);
        time_t et = time(0);

        cout << "Computed in " << et-st << " seconds." << endl;

        Y = tr.getY();
        I = tr.getI();
        Q = tr.getQ();
        G = tr.getG();

        epp::vector<double> Yerr = tr.getYerr();
        epp::vector<double> Qerr = tr.getQerr();


        epp::vector<double> II;
        epp::vector<double> YI;
        epp::vector<double> QI;
        epp::vector<double> GI;

        if(invcheck)
        {
            cout << "Inversion check..." << endl;
            time_t st = time(0);
            tr.transform(inversion ? tr.FtoR : tr.RtoF , twodim);
            time_t et = time(0);
            cout << "Computed in " << et-st << " seconds." << endl;

            if(inversion)
            {
                QI = tr.getQ();
                if(twodim)
                    GI = tr.getG();
            }
            else
            {
                YI = tr.getY();
                if(twodim)
                    II = tr.getI();
            }

        }

        epp::vector<double> QS(T);
        epp::vector<double> GS(T);
        epp::vector<double> QSe(T);
        epp::vector<double> GSe(T);

        if(simcheck)
        {
            cout << "Simulation check, n=" << simn << endl;
            time_t st = time(0);

            epp::vector<double> sumQ=zero_vector(T);
            epp::vector<double> sumG=zero_vector(T);
            epp::vector<double> ssqQ=zero_vector(T);
            epp::vector<double> ssqG=zero_vector(T);

            for(int i=0; i<simn; i++)
            {
                cout << ".";
                epp::vector<double> q(T);
                epp::vector<double> g(T);
                tr.sim(&q, twodim ? &g : 0);
                sumQ += q;
                for(unsigned j=0; j<T; j++)
                    ssqQ[j] += q[j]*q[j];
                if(twodim)
                {
                    sumG += g;
                    for(unsigned j=0; j<T; j++)
                        ssqG[j] += g[j]*g[j];
                }
            }
            QS = sumQ / simn;
            if(twodim)
                GS = sumG / simn;
            cout << endl;

            for(unsigned i=0; i<T; i++)
            {
                QSe[i] = sqrt(ssqQ[i] / (simn*simn) + QS[i]*QS[i] / simn);
                if(twodim)
                    GSe[i] = sqrt(ssqG[i] / (simn*simn) + GS[i]*GS[i] / simn);
            }

            time_t et = time(0);
            cout << "Computed in " << et-st << " seconds." << endl;

        }

        out << "Y,I,Q,G,N,Ystar,Ye,Qe,";
        if(YI.size())
            out << "YI" << ",";
        if(II.size())
            out << "II" << ",";
        if(QI.size())
            out << "QI" << ",";
        if(GI.size())
            out << "GI" << ",";
        if(simcheck)
        {
            out << "QS,QSe,";
            if(twodim)
                out << "GS,GSe,";
        }

        out << "M,S" << endl;

        epp::vector<double> Ystar(T);
        if(!multig)
            Ystar = tr.getYstar();
        for(unsigned i=0; i<T; i++)
        {
            out << Y[i] << ",";
            out << I[i] << ",";
            out << Q[i] << ",";
            out << G[i] << ",";
            out << tr.getnc()[i] << ",";
            if(multig)
                out << "NA,";
            else
                out << Ystar[i] << ",";
            out << Yerr[i] << ",";
            out << Qerr[i] << ",";
            if(YI.size())
                out << YI[i] << ",";
            if(II.size())
                out << II[i] << ",";
            if(QI.size())
                out << QI[i] << ",";
            if(GI.size())
                out << GI[i] << ",";
            if(simcheck)
            {
                out << QS[i] << "," << QSe[i] << ",";
                if(twodim)
                    out << GS[i] << "," << GSe[i] << ",";
            }
            out << tr.getlogWmean(i) << ",";
            out << tr.getlogWsigma(i) << endl;
        }
        cout << "Output written to " << output << endl;
    }
    catch (std::exception& e)
    {
        std::cout << "There was an error: " <<  std::endl
           << e.what() << std::endl;
        std::cout << "For help, use ? as argument." << endl;
        return 1;
    }
    return 0;
}

// todo
// - zprovoznit errors
