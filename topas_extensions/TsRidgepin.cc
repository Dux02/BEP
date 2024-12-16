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
#include "G4LogicalVolume.hh"
#include "G4PVReplica.hh"

#include "globals.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"

#include "math.h"
#include <algorithm>

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
    fRidgepinBaseWidth = fPm->GetDoubleParameter(GetFullParmName("RidgepinBaseWidth"), "Length");
    fRidgepinSpacing = fPm->GetDoubleParameter(GetFullParmName("RidgepinSpacing"), "Length");
    // fMaterial = fPm->GetStringParameter(GetFullParmName("Material"));
    fRidgeFilterBaseHeight = fPm->GetDoubleParameter(GetFullParmName("RidgeFilterBaseHeight"), "Length");
    fRidgepinElements = fPm->GetIntegerParameter(GetFullParmName("RidgepinElements"));
    fRidgeFilterNumberOfPinsX = fPm->GetIntegerParameter(GetFullParmName("RidgeFilterNumberOfPinsX"));
    fRidgeFilterNumberOfPinsY = fPm->GetIntegerParameter(GetFullParmName("RidgeFilterNumberOfPinsY"));

    fRidgepinWidths = fPm-> GetDoubleVector(GetFullParmName("RidgepinWidths"), "Length");
}

G4VPhysicalVolume* TsRidgepin::Construct(){
        BeginConstruction();

    // Materials
    G4NistManager* man = G4NistManager::Instance();
    G4Material* Air = man->FindOrBuildMaterial("G4_AIR");
    G4Material* Water = man->FindOrBuildMaterial("G4_WATER");
    // Gets maximum width.
    fRidgePinMaxWidth = fRidgepinWidths[0];

    if (fRidgePinMaxWidth > fRidgepinBaseWidth){
        G4cerr << "Topas is exiting due to a serious error in geometry setup." << G4endl;
        G4cerr << "The maximum ridgepin width is greater than the base width" << G4endl;
        fPm->AbortSession(1);
    }

    // Initializing paramaters
    fRidgePinHeightStep = fRidgepinMaxHeight / fRidgepinElements;

    fRidgeFilterWidth_X = ((fRidgeFilterNumberOfPinsX) * fRidgepinSpacing) + ((fRidgeFilterNumberOfPinsX) * fRidgepinBaseWidth);
    fRidgeFilterWidth_Y = ((fRidgeFilterNumberOfPinsY) * fRidgepinSpacing) + ((fRidgeFilterNumberOfPinsY) * fRidgepinBaseWidth);
    
    //Constructing a World for the Ridgepin to be in
    world_HX = fRidgeFilterWidth_X;
    world_HY = fRidgeFilterWidth_Y;
    world_HZ = fRidgepinMaxHeight + fRidgeFilterBaseHeight;
    G4Box* worldBox = new G4Box("World", world_HX, world_HY, world_HZ);
    G4LogicalVolume* worldLog = CreateLogicalVolume("WorldLog", worldBox);
    G4VPhysicalVolume* worldPhys = CreatePhysicalVolume(worldLog);
    // G4String fMaterial = "G4_Water";
    // Constructing the ridgepin
    G4ThreeVector* position_base = new G4ThreeVector(0, 0, -(world_HZ) );
    G4RotationMatrix* rotation_base = new G4RotationMatrix();
    G4VSolid* baseLayer = new G4Box("baseLayer", world_HX, world_HY, 0.5*fRidgeFilterBaseHeight);
    G4LogicalVolume* baseLayerLog = CreateLogicalVolume("baseLayerLog", baseLayer);
    G4VPhysicalVolume* baseLayerPhys = CreatePhysicalVolume("baselayerPhys", baseLayerLog, rotation_base, position_base, worldPhys);

    for (G4int x = 0; x < fRidgeFilterNumberOfPinsX; x++){
        for (G4int y  = 0; y < fRidgeFilterNumberOfPinsY; y++){
            for (G4int i = 0; i < fRidgepinElements; i++){
                G4double posX = (x * fRidgepinSpacing) + (2 * x * fRidgepinBaseWidth) - world_HX + 0.5*(2*fRidgepinSpacing+fRidgePinMaxWidth);
                G4double posY = (y * fRidgepinSpacing) + (2 * y * fRidgepinBaseWidth) - world_HY + 0.5*(2*fRidgepinSpacing+fRidgePinMaxWidth);
                G4double posZ = ((i) * ( 2 * fRidgePinHeightStep) - world_HZ + 0.5*fRidgeFilterBaseHeight + fRidgeFilterBaseHeight);
                G4ThreeVector* position = new G4ThreeVector(posX, posY, posZ);   
                G4String fName = "Ridgepin_x:"+ std::to_string(x) + "_y:"+ std::to_string(y) + "_z:" + std::to_string(i) ;
                G4RotationMatrix* rotation = new G4RotationMatrix();
                // Creating the shape of the ridgepin layer
                G4Trd* ridgepin_layer = new G4Trd(fName, fRidgepinWidths[i], fRidgepinWidths[i+1], fRidgepinWidths[i], fRidgepinWidths[i+1], fRidgePinHeightStep);
                // Creating the logical volume of the ridgepin layer
                G4LogicalVolume* ridgepinLog = CreateLogicalVolume("ridgepinLog", ridgepin_layer);
                // Creating the physical volume of the ridgepin layer
                G4VPhysicalVolume* ridgepinPhys = CreatePhysicalVolume(fName, ridgepinLog, rotation, position, worldPhys);
            }
        }
    }
}