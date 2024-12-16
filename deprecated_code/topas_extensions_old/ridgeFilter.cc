// Component for MyRidgeFilter

#include "TsRidgeFilter.hh"

#include "TsParameterManager.hh"

#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4TessellatedSolid.hh"
#include "G4QuadrangularFacet.hh"
#include "G4SystemOfUnits.hh"

TsRidgeFilter::TsRidgeFilter(TsParameterManager* pM, TsExtensionManager* eM, TsMaterialManager* mM, TsGeometryManager* gM,
                             TsVGeometryComponent* parentComponent, G4VPhysicalVolume* parentVolume, G4String& name)
: TsVGeometryComponent(pM, eM, mM, gM, parentComponent, parentVolume, name)
{;}


TsRidgeFilter::~TsRidgeFilter()
{;}


G4VPhysicalVolume* TsRidgeFilter::Construct()
{
    BeginConstruction();

    const G4int n_z  = fPm->GetVectorLength(GetFullParmName("ZPoints"));
    G4double*     z  = fPm->GetDoubleVector(GetFullParmName("ZPoints"), "Length");
    const G4int n_x  = fPm->GetVectorLength(GetFullParmName("XPoints"));
    G4double*     x  = fPm->GetDoubleVector(GetFullParmName("XPoints"), "Length");
    const G4double width  = fPm->GetDoubleParameter(GetFullParmName("Width"), "Length");
    G4double*   pos_x     = fPm->GetDoubleVector(GetFullParmName("Displacement"), "Length");
    const G4int n_pos_x   = fPm->GetVectorLength(GetFullParmName("Displacement"));

    // for the ridge array and whole box
    G4double z_max = 0.0;
    for (int i = 0; i < n_z; ++i)
    {
        if  (z_max <= z[i])
            z_max = z[i];
    }

    G4double x_max = 0.0;
    G4double x_min = 0.0;
    for (int i = 0; i < n_pos_x; ++i)
    {
        if (x_max <= pos_x[i])
            x_max = pos_x[i];

        if (x_min >= pos_x[i])
            x_min = pos_x[i];
    }

    G4double hrx = 0.5*(x_max  - x_min + width);
    G4double hry = 0.5 * fPm->GetDoubleParameter(GetFullParmName("Length"), "Length");
    G4double hrz = 0.5 * z_max;

    //================================================================================
    // Geometry setup
    //================================================================================
    // Whole Box
    G4String envelopeMaterialName = fParentComponent->GetResolvedMaterialName();
    G4Box* svWholeBox = new G4Box("RidgeBox", hrx, hry, hrz);
    fEnvelopeLog = CreateLogicalVolume("RidgeBox", envelopeMaterialName, svWholeBox);
    fEnvelopeLog->SetVisAttributes(fPm->GetInvisible());
    fEnvelopePhys = CreatePhysicalVolume(fEnvelopeLog);

    //Build a Ridge solid
    G4TessellatedSolid* sv_ridge_i = new G4TessellatedSolid("Ridge");

    if ( fPm->ParameterExists(GetFullParmName("PrintInformation")) && fPm->GetBooleanParameter(GetFullParmName("PrintInformation")) ) {
        G4cout<<" Ridge points (x,z) ---   :"<<  n_x  <<G4endl;
        G4cout<<"       P initial : (0, 0) cm" <<G4endl;
    }

    //Iterate user input values in vector

    G4double offset = 0.0;

    for (int i = 0; i < (n_x -1); ++i  ) {
        if ( fPm->ParameterExists(GetFullParmName("PrintInformation")) && fPm->GetBooleanParameter(GetFullParmName("PrintInformation")) ) {
            G4cout<<"       P "<<i<<"th     : ("<< x[i]/cm <<", " << z[i]/cm << ") cm" <<G4endl;
        }

        // preconditions
        // Width = 2*max(x);
        // Width and length need to be the same value
        // x[0] = 0;
        // z[0] = 0;

        // left bottom
        // left top
        // right top
        // right bottom
        // x, y, z
        G4ThreeVector *tl1 = new G4ThreeVector(x[i], 			width - offset, z[i]);
        G4ThreeVector *tr1 = new G4ThreeVector(width - offset,	width - offset, z[i]);
        G4ThreeVector *br1 = new G4ThreeVector(width - offset, x[i],   		z[i]);
        G4ThreeVector *bl1 = new G4ThreeVector(x[i], 			x[i],   		z[i]);

        offset = x[i+1];

        G4ThreeVector *tl2 = new G4ThreeVector(x[i+i], 		    width - offset, z[i+i]);
        G4ThreeVector *tr2 = new G4ThreeVector(width - offset,  width - offset, z[i+1]);
        G4ThreeVector *br2 = new G4ThreeVector(width - offset,  x[i+1], 		z[i+1]);
        G4ThreeVector *bl2 = new G4ThreeVector(x[i+1], 		    x[i+1], 		z[i+1]);


        G4TriangularFacet *trapTop   = new G4QuadrangularFacet(tl2, tr2, tr1, tl1, ABSOLUTE);
        G4TriangularFacet *trapRight = new G4QuadrangularFacet(tr2, br2, br1, tr1, ABSOLUTE);
        G4TriangularFacet *trapBot   = new G4QuadrangularFacet(br2, bl2, bl1, br1, ABSOLUTE);
        G4TriangularFacet *trapLeft  = new G4QuadrangularFacet(bl2, tl2, tl1, bl1, ABSOLUTE);

        sv_ridge_i->AddFacet((G4VFacet*) trapTop);
        sv_ridge_i->AddFacet((G4VFacet*) trapRight);
        sv_ridge_i->AddFacet((G4VFacet*) trapBot);
        sv_ridge_i->AddFacet((G4VFacet*) trapLeft);
    }


    // bottom face
    tlb = G4ThreeVector(0, 	   width, 0);
    trb = G4ThreeVector(width, width, 0);
    brb = G4ThreeVector(width, 0,     0);
    blb = G4ThreeVector(0,	   0, 	  0);


    // top face
    int j = n_x-1;
    tlt = G4ThreeVector(x[j], 	      width - x[j], z[j]);
    trt = G4ThreeVector(width - x[j], width - x[j], z[j]);
    brt = G4ThreeVector(width - x[j], x[j],         z[j]);
    blt = G4ThreeVector(x[j],         x[j], 	    z[j]);

    G4TriangularFacet *botf = new G4QuadrangularFacet(blb, brb, trb, tlb, ABSOLUTE);
    G4TriangularFacet *topf = new G4QuadrangularFacet(blt, brt, trt, tlt, ABSOLUTE);

    sv_ridge_i->AddFacet((G4VFacet*) botf);
    sv_ridge_i->AddFacet((G4VFacet*) topf);



    if ( fPm->ParameterExists(GetFullParmName("PrintInformation")) && fPm->GetBooleanParameter(GetFullParmName("PrintInformation")) ) {
        G4cout<<"       P "<< n_x-1 <<"th     : ("<< x[n_x-1]/cm <<", " << z[n_x-1]/cm << ") cm" <<G4endl;
        G4cout<<"       P final : ("<< width/cm <<", 0) cm" <<G4endl;

    }

    sv_ridge_i->SetSolidClosed(true);
    G4LogicalVolume* lvRidgeUnit = CreateLogicalVolume(sv_ridge_i);

    //Ridge placement
    for (G4int i = 0; i < n_pos_x; ++i)
    {
        CreatePhysicalVolume("Ridge", i, true, lvRidgeUnit, 0, new G4ThreeVector(-width/2.0 + pos_x[i], 0., -hrz), fEnvelopePhys);
    }
    return fEnvelopePhys;
}
