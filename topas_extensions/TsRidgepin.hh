#ifndef TsRidgepin_hh
#define TsRidgepin_hh

#include "TsVGeometryComponent.hh"

class TsRidgepin : public TsVGeometryComponent
{
public:
        TsRidgepin(TsParameterManager* pM, TsExtensionManager* eM, TsMaterialManager* mM, TsGeometryManager* gM,
                TsVGeometryComponent* parentComponent, G4VPhysicalVolume* parentVolume, G4String& name);
        ~TsRidgepin();

        G4VPhysicalVolume* Construct();
    void ResolveParameters();
private:
    G4double fRidgepinMaxHeight;
    G4double fRidgepinSpacing;
    G4String fMaterial;
    G4double fRidgeFilterBaseHeight, fRidgepinBaseWidth;
    G4double* fRidgepinWidths;

    G4int fRidgeFilterNumberOfPinsX;
    G4int fRidgeFilterNumberOfPinsY;
    G4int fRidgepinElements;
    G4double world_HX, world_HY, world_HZ, fRidgeFilterWidth_X, fRidgeFilterWidth_Y, fRidgePinMaxWidth, fRidgePinHeightStep;
};

#endif