class Visit:
    # ENROLLHD visit

    #def __init__(self, seq, age, days, vtype, tfc, motor, function, cognitive, pbas):
        # self.seq = seq              # visit sequence nr
        # self.age = age              # Age at time of visit
        # self.days = days            # Days since Baseline visit
        # self.vtype = vtype          # Visit type
        # self.motor = motor          # UHDRS Motor form
        # self.tfc = tfc              # UHDRS TFC form
        # self.function = function    # UHDRS Function form
        # self.cognitive = cognitive  # UHDRS Cognitive form
        # self.pbas = pbas            # PBA-s form

    #def __init__(self, seq, age, visit, hdcat, motscore, miscore, tfcscore, fascore, fiscore, sdmt1, sdmt2, sit1, sit2,
    #             sit3, depscore, irascore, psyscore, aptscore, exfscore):
#
    #    self.seq = seq
    #    self.age = age
    #    self.visit = visit
    #    self.hdcat = hdcat
    #    self.motscore = motscore
    #    self.miscore = miscore
    #    self.tfcscore = tfcscore
    #    self.fascore = fascore
    #    self.fiscore = fiscore
    #    self.sdmt1 = sdmt1
    #    self.sdmt2 = sdmt2
    #    self.sit1 = sit1
    #    self.sit2 = sit2
    #    self.sit3 = sit3
    #    self.depscore = depscore
    #    self.irascore = irascore
    #    self.psyscore = psyscore
    #    self.aptscore = aptscore
    #    self.exfscore = exfscore
    
    def __init__(self):
        self.seq = -1
        self.age = -1
        self.visit = -1
        self.hdcat = -1
        self.motscore = -1
        self.miscore = -1
        self.tfcscore = -1
        self.fascore = -1 
        self.fiscore = -1
        self.sdmt1 = -1
        self.sdmt2 = -1
        self.sit1 = -1
        self.sit2 = -1
        self.sit3 = -1
        self.depscore = -1
        self.irascore = -1
        self.psyscore = -1
        self.aptscore = -1
        self.exfscore = -1


class Motor:
    # UHDRS Motor Diagnostic  Confidence (Motor)

    def __init__(self,motscore, miscore, ocularh, ocularv, sacinith, sacinitv, sacvelh, sacvelv, dysarth, tongue, fingtapr, fingtapl, prosupr, prosupl,
                 luria, rigarmr, rigarml, brady, dysttrnk, dystrue, dystlue, dystrle, dystlle, chorface, chorbol, chortrnk, chorrue, chorlue, chorrle,
                 chorlle, gait, tandem, retropls, diagconf):

        # General Scores
        self.motscore = motscore    # UHDRS motor score
        self.miscore = miscore      # UHDRS motor score (incomplete)

        # Group Ocular Pursuit

        self.ocularh = ocularh  # Horizontal
        self.ocularv = ocularv  # Vertical

        # Group Saccade initiation

        self.sacinith = sacinith    # Horizontal
        self.sacinitv = sacinitv    # Vertical

        # Group Saccade velocity

        self.sacvelh = sacvelh  # Horizontal
        self.sacvelv = sacvelv  # Vertical
        self.dysarth = dysarth  # Dysarthria
        self.tongue = tongue    # Tongue protrusion

        # Group Finger taps

        self.fingtapr = fingtapr    # Right
        self.fingtapl = fingtapl    # Left

        # Group Pronate supinate‐hand

        self.prosupr = prosupr  # Right
        self.prosupl = prosupl  # Left
        self.luria = luria      # Luria

        # Group Rigidity‐arms

        self.rigarmr = rigarmr  # Right
        self.rigarml = rigarml  # Left
        self.brady = brady      # Bradykinesia-Body

        # Group Maximal dystonia

        self.dysttrnk = dysttrnk    # Trunk
        self.dystrue = dystrue      # RUE -
        self.dystlue = dystlue      # LUE
        self.dystrle = dystrle      # RLE
        self.dystlle = dystlle      # LLE

        # Group Maximal chorea

        self.chorface = chorface    # Face
        self.chorbol = chorbol      # BOL
        self.chortrnk = chortrnk    # Trunk
        self.chorrue = chorrue      # RUE
        self.chorlue = chorlue      # LUE
        self.chorrle = chorrle      # RLE
        self.chorlle = chorlle      # LLE
        self.gait = gait            # Gait
        self.tandem = tandem        # Tandem walking
        self.retropls = retropls    # Retropulsion pull test

        # Diagnostic Confidence

        self.diagconf = diagconf

class TFC:
    # UHDRS Total Functional Capacity (TFC)

    def __init__(self, tfcscore, occupatn, finances, chores, adl, carelvl):

        self.tfcsore = tfcscore     # Functional Score
        self.occupatn = occupatn    # Occupation
        self.finances = finances    # Finances
        self.chores = chores        # Domestic chores
        self.adl = adl              # ADL
        self.carelvl = carelvl      # Care level


class Function:
    # UHDRS Functional Assessment Independence Scale (Function)

    def __init__(self, fascore, fiscore, emplusl, emplany, volunt, fafinan, grocery, cash, supchild, drive, housewrk,
                 laundry, prepmeal, telephon, ownmeds, feedself, dress, bathe, pubtrans, walknbr, walkfall, walkhelp,
                 comb, trnchair, bed, toilet, carehome, indepscl):

        self.fascore = fascore
        self.fiscore = fiscore
        self.emplusl = emplusl
        self.emplany = emplany
        self.volunt = volunt
        self.fafinan = fafinan
        self.grocery = grocery
        self.cash = cash
        self.supchild = supchild
        self.drive = drive
        self.housewrk = housewrk
        self.laundry = laundry
        self.prepmeal = prepmeal
        self.telephon = telephon
        self.ownmeds = ownmeds
        self.feedself = feedself
        self.dress = dress
        self.bathe = bathe
        self.pubtrans = pubtrans
        self.walknbr = walknbr
        self.walkfall = walkfall
        self.walkhelp = walkhelp
        self.comb = comb
        self.trnchair = trnchair
        self.bed = bed
        self.toilet = toilet
        self.carehome = carehome
        self.indepscl = indepscl


class Cognitive:
    # Cognitive Assessments (Cognitive)

    def __init__(self, gen1, gen2, gen3, gen4, gen5, gen6):

    # Section Specifics

        self.gen1 = gen1
        self.gen2 = gen2
        self.gen3 = gen3
        self.gen4 = gen4
        self.gen5 = gen5
        self.gen6 = gen6



