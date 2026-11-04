#include "flow.hpp"
#include <algorithm>
#include <cmath>

namespace bld::wounds
{
    float advance(source &w,float dt)
    {
        if(!std::isfinite(dt) || dt<=0.f ||
           w.remaining_ml<=0.f || w.initial_rate_ml_s<=0.f)return 0.f;
        dt=std::min(dt,.15f);
        const float decay=std::max(.001f,w.decay_seconds);
        const float start=std::exp(-w.age/decay);
        const float end=std::exp(-(w.age+dt)/decay);
        float volume=std::max(0.f,w.initial_rate_ml_s)*decay*(start-end);
        // A bounded pressure pulsation can be game- or anatomy-driven.
        // The baseline non-pulsatile source is used unless specified.
        if(w.pulse_hz>0.f)
        {
            constexpr float pi=3.14159265358979323846f;
            volume*=1.f+.3f*std::sin(2.f*pi*w.pulse_hz*
                (w.age+.5f*dt));
        }
        volume=std::clamp(volume,0.f,w.remaining_ml);
        w.remaining_ml-=volume;
        w.age+=dt;
        return volume;
    }
}
