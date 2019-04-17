import visit


class Person:

    def __init__(self, subjid, state):
        self.subjid = subjid
        self.visit_list = []
        self.state = state
        self.visits = 0  # Total visits
        self.visits_ok = 0  # visits excluding missing/phone
        self.region = -1  # 0 for EU, 1 for LatinAm, 2 for NA, 3 for Australasia
        self.sex = -1 # 0 for female, 1 for male
        self.race = -1
        self.hxsid = -1
        self.momhd = -1
        self.dadhd = -1
        self.momagesx = -1
        self.dadagesx = -1
        self.caghigh = -1
        self.caglow = -1
        self.fhx = -1

    def add_visit(self, v):
        self.visit_list.append(v)
        self.visits = len(self.visit_list)  # Total visits
        self.state = v.hdcat

    def add_info(self, region, sex, race, hxsid, caghigh, caglow, momhd, momagesx, dadhd, dadagesx, fhx):
        self.region = region
        self.sex = 0 if sex == 'f' else 1
        self.race = race
        self.hxsid = hxsid
        self.caghigh = caghigh
        self.caglow = caglow
        self.momhd = momhd
        self.dadhd = dadhd
        self.fhx = fhx
        self.dadagesx = dadagesx if dadhd == 1 else 0
        self.momagesx = momagesx if momhd == 1 else 0



