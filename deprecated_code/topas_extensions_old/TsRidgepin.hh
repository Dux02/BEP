#ifndef TsRidgepin_hh
#define TsRidgepin_hh

#include "TsVGeometryComponent.hh"

class TsRidgepin : public TsVGeometryComponent
{
public:
    TsRidgepin(TsParameterManager* pM, TsExtensionManager* eM, TsMaterialManager* mM, TsGeometryManager* gM,
               TsVGeometryComponent* parentComponent, G4VPhysicalVolume* parentVolume, G4String& name);
    virtual ~TsRidgepin();

    virtual G4VPhysicalVolume* Construct() override;
    virtual void ResolveParameters() override;
private:
    G4double fRidgepinMaxHeight;
    std::vector<G4double> fRidgepinWidths;
    G4double fRidgepinSpacing;
    G4String fRidgePinMaterial;
    G4double fRidgeFilterBaseHeight;
    G4int fRidgeFilterNumberOfPinsX;
    G4int fRidgeFilterNumberOfPinsY;
};

#endif