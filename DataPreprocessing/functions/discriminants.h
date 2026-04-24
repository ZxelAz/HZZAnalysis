#ifndef DISCRIMINANTS_H
#define DISCRIMINANTS_H

#include <cassert>
#include <memory>
#include <string>
#include "TFile.h"
#include "TSpline.h"

// Minimal reimplementation of the original cConstantSpline helper
class cConstantSpline {
public:
    explicit cConstantSpline(const std::string& filename) : filename_(filename), spline_(nullptr) {}

    double eval(double ZZMass, bool /*isDbkg*/ = false) {
        initspline();
        return spline_->Eval(ZZMass);
    }

private:
    void initspline() {
        if (!spline_) {
            TFile* f = TFile::Open(filename_.c_str());
            spline_.reset(static_cast<TSpline3*>(f->Get("sp_gr_varReco_Constant_Smooth")->Clone()));
            f->Close();
        }
        assert(spline_.get());
    }

    std::string filename_;
    std::unique_ptr<TSpline3> spline_;
};

// Location of the ME KD spline ROOT files; overridden from Python via
//   ROOT.gInterpreter.Declare('#define HZZ_CCONST_DIR "/abs/path/to/cconstants"')
// before this header is included. Falls back to a relative "cconstants" dir.
#ifndef HZZ_CCONST_DIR
#define HZZ_CCONST_DIR "cconstants"
#endif

namespace {
    cConstantSpline DVBF2jetsSpline(HZZ_CCONST_DIR "/SmoothKDConstant_m4l_DjjVBF13TeV.root");
    cConstantSpline DVBF1jetSpline(HZZ_CCONST_DIR "/SmoothKDConstant_m4l_DjVBF13TeV.root");
    cConstantSpline DZHhSpline(HZZ_CCONST_DIR "/SmoothKDConstant_m4l_DjjZH13TeV.root");
    cConstantSpline DWHhSpline(HZZ_CCONST_DIR "/SmoothKDConstant_m4l_DjjWH13TeV.root");
}

extern "C" float getDVBF2jetsConstant(float ZZMass){
    return static_cast<float>(DVBF2jetsSpline.eval(ZZMass, false));
}
extern "C" float getDVBF1jetConstant(float ZZMass){
    return static_cast<float>(DVBF1jetSpline.eval(ZZMass, false));
}
extern "C" float getDWHhConstant(float ZZMass){
    return static_cast<float>(DWHhSpline.eval(ZZMass, false));
}
extern "C" float getDZHhConstant(float ZZMass){
    return static_cast<float>(DZHhSpline.eval(ZZMass, false));
}

// Discriminant helpers
float DVBF2j_ME(float p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal,
                float p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal,
                float ZZMass) {
    float c_Mela2j = getDVBF2jetsConstant(ZZMass);
    return 1.f/(1.f + c_Mela2j * p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal / p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal);
}

float DVBF1j_ME(float p_JVBF_SIG_ghv1_1_JHUGen_JECNominal,
                float pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal,
                float p_JQCD_SIG_ghg2_1_JHUGen_JECNominal,
                float ZZMass) {
    float c_Mela1j = getDVBF1jetConstant(ZZMass);
    return 1.f/(1.f + c_Mela1j * p_JQCD_SIG_ghg2_1_JHUGen_JECNominal / (p_JVBF_SIG_ghv1_1_JHUGen_JECNominal * pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal));
}

float DWHh_ME(float p_HadWH_SIG_ghw1_1_JHUGen_JECNominal,
              float p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal,
              float p_HadWH_mavjj_JECNominal,
              float p_HadWH_mavjj_true_JECNominal,
              float ZZMass) {
    float c_MelaWH = getDWHhConstant(ZZMass);
    return 1.f/(1.f + c_MelaWH * (p_HadWH_mavjj_true_JECNominal * p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal) / (p_HadWH_mavjj_JECNominal * p_HadWH_SIG_ghw1_1_JHUGen_JECNominal));
}

float DZHh_ME(float p_HadZH_SIG_ghz1_1_JHUGen_JECNominal,
              float p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal,
              float p_HadZH_mavjj_JECNominal,
              float p_HadZH_mavjj_true_JECNominal,
              float ZZMass) {
    float c_MelaZH = getDZHhConstant(ZZMass);
    return 1.f/(1.f + c_MelaZH * (p_HadZH_mavjj_true_JECNominal * p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal) / (p_HadZH_mavjj_JECNominal * p_HadZH_SIG_ghz1_1_JHUGen_JECNominal));
}

#endif // DISCRIMINANTS_H
