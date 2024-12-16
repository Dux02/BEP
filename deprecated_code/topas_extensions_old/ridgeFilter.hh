#ifndef TS_RIDGE_FILTER_HH
#define TS_RIDGE_FILTER_HH

#include "TsVGeometryComponent.hh"
#include "G4VPhysicalVolume.hh"

// Forward declarations
class TsParameterManager;
class TsExtensionManager;
class TsMaterialManager;
class TsGeometryManager;

class TsRidgeFilter : public TsVGeometryComponent
{
public:
    // Constructor
    TsRidgeFilter(TsParameterManager* pM, TsExtensionManager* eM, TsMaterialManager* mM, TsGeometryManager* gM,
                  TsVGeometryComponent* parentComponent, G4VPhysicalVolume* parentVolume, G4String& name);

    // Destructor
    virtual ~TsRidgeFilter();

    // Main construction function
    virtual G4VPhysicalVolume* Construct() override;

private:
    // Helper functions or private variables can be added here if needed
};

#endif // TS_RIDGE_FILTER_HH
