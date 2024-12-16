// Component for TsRidgepin

#include "TsRidgepin.hh"

#include "TsParameterManager.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Trd.hh"
#include "G4Box.hh"

#include "G4SystemOfUnits.hh"
#include "G4PVPlacement.hh"
#include "G4Color.hh"
#include "G4PhysicalConstants.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

TsRidgepin::TsRidgepin(TsParameterManager* pM, TsExtensionManager* eM, TsMaterialManager* mM, TsGeometryManager* gM,
                       TsVGeometryComponent* parentComponent, G4VPhysicalVolume* parentVolume, G4String& name)
: TsVGeometryComponent(pM, eM, mM, gM, parentComponent, parentVolume, name)
{
    ResolveParameters();
}

TsRidgepin::~TsRidgepin()
{;}


void TsRidgepin::ResolveParameters() {
    fRidgepinMaxHeight = fPm->GetDoubleParameter(GetFullParmName("RidgepinMaxHeight"), "Length");
    fRidgepinWidths = fPm->GetVectorDouble(GetFullParmName("RidgepinWidths"), "Length");
    fRidgepinSpacing = fPm->GetDoubleParameter(GetFullParmName("RidgepinSpacing"), "Length");
    fRidgePinMaterial = fPm->GetStringParameter(GetFullParmName("RidgepinMaterial"));
    fRidgeFilterBaseHeight = fPm->GetDoubleParameter(GetFullParmName("RidgeFilterBaseHeight"), "Length");

    fRidgeFilterNumberOfPinsX = fPm->GetIntegerParameter(GetFullParmName("RidgeFilterNumberOfPinsX"));
    fRidgeFilterNumberOfPinsY = fPm->GetIntegerParameter(GetFullParmName("RidgeFilterNumberOfPinsY"));
    

    // Calculated paramters
    fRidgePinMaxWidth = *std::max_element(fRidgepinWidths.begin(), fRidgepinWidths.end());
    fRidgePinElements = fRidgepinWidths.size();
    fRidgePinHeightStep = fRidgepinMaxHeight / fRidgePinElements;

    fRidgeFilterWidth_X = ((fRidgeFilterNumberOfPinsX - 1) * fRidgepinSpacing) + (2 * fRidgeFilterNumberOfPinsX * fRidgepinWidths[0]);
    fRidgeFilterWidth_Y = ((fRidgeFilterNumberOfPinsY - 1) * fRidgepinSpacing) + (2 * fRidgeFilterNumberOfPinsY * fRidgepinWidths[0]);
}

G4VPhysicalVolume* TsRidgepin::Construct(){
    BeginConstruction();
    //Constructing a World for the Ridgepin to be in
    G4double world_HX = fRidgePinMaxWidth;
    G4double world_HY = fRidgePinMaxWidth;
    Gydouble world_HZ = fRidgepinMaxHeight;
    G4Box* worldBox = new G4Box("World", world_HX, world_HY, world_HZ);

    G4LogicalVolume* worldLog = new G4LogicalVolume(worldBox, fMM->GetMaterial("G4_AIR"), "World");
    // Constructing the ridgepin

    G4Vector G4RidgepinWidths = fMM->GetVector(fRidgepinWidths);
    G4Double HZ = fRidgePinHeightStep; 

    for (int = 1; i <= fRidgePinElements; i++){
        G4ThreeVector position = G4ThreeVector(0, 0, (i-1) * HZ);   
        G4String fName = "Ridgepin_" + std::to_string(i);
        // Creating the shape of the ridgepin layer
        G4Trd* ridgepin = new G4Trd(fName, G4RidgepinWidths[i], G4RidgepinWidths[i+1], G4RidgepinWidths[i], G4RidgepinWidths[i+1], HZ);
        // Creating the logical volume of the ridgepin layer
        G4LogicalVolume* ridgepinLog = new CreateLogicalVolume(fName, fRidgePinMaterial, ridgepin);
        // Creating the physical volume of the ridgepin layer
        G4VPhysicalVolume* ridgepinPhys = new G4PVPlacement(0, position, ridgepinLog, fName, worldLog, false, 0);
        }
        
}