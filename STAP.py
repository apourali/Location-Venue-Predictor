"""
User Activity Preference by Leveraging User Spatial Temporal Characteristics in LBSNs

Remodeled by: Alireza Pourali, M.A.Sc Candidate of Computer and Electrical Engineering at Ryerson University
Reference:
http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf
"""


#IGNORE WARNINGS
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import math as mt
from scipy import stats
from sklearn.cluster import KMeans
from datetime import datetime
from difflib import SequenceMatcher

#----------Nearest PFR to a check-in--------------------
def nearest_cluster(checkinLO,checkinLA):
    checkinLO = float(checkinLO)
    checkinLA = float(checkinLA)
    dist = mt.sqrt((x_means.cluster_centers_[0][0] - checkinLO) ** 2 + (x_means.cluster_centers_[0][1] - checkinLA) ** 2)
    index = 0
    x = 0
    y = 0
    for i in x_means.cluster_centers_:
        distNew = mt.sqrt((x_means.cluster_centers_[x][0] - checkinLO) ** 2 + (x_means.cluster_centers_[y][1] - checkinLA) ** 2)
        if (distNew < dist):
            index = x

        x+=1
        y+=1
    return index


#---------Venue Categories Crawled from Foursquare API-------------
categoriez = []
with open('C:\Users\Alireza\Desktop\Datasets\New York\VenueCategories.txt') as inputfile:
        for line in inputfile:
            categoriez.append((line.split('\t'))[0])
            categoriez.append((line.split('\t'))[1])

categoriez = np.array(categoriez)
categoriez = np.reshape(categoriez, (-1, 2))

#--------------Optimzed results------------------------------------
def optimized(a, b):
    return SequenceMatcher(None, a, b).ratio()

#--------------Categories and Values Sorter------------------------
def categories(centroidsCategories):

    categoriesDict = {'/arts_entertainment/aquarium_':centroidsCategories[2],
                        '/arts_entertainment/arcade_':centroidsCategories[3],
                        '/arts_entertainment/artgallery_':centroidsCategories[4],
                        '/arts_entertainment/billiards_':centroidsCategories[5],
                        '/arts_entertainment/bowling_':centroidsCategories[6],
                        '/arts_entertainment/casino_':centroidsCategories[7],
                        '/arts_entertainment/comedyclub_':centroidsCategories[8],
                        '/arts_entertainment/default_':centroidsCategories[9],
                        '/arts_entertainment/historicsite_':centroidsCategories[10],
                        '/arts_entertainment/movietheater_':centroidsCategories[11],
                        '/arts_entertainment/museum_':centroidsCategories[12],
                        '/arts_entertainment/musicvenue_':centroidsCategories[13],
                        '/arts_entertainment/performingarts_':centroidsCategories[14],
                        '/arts_entertainment/racetrack_':centroidsCategories[15],
                        '/arts_entertainment/stadium_':centroidsCategories[16],
                        '/arts_entertainment/themepark_':centroidsCategories[17],
                        '/arts_entertainment/waterpark_':centroidsCategories[18],
                        '/arts_entertainment/zoo_':centroidsCategories[19],
                        '/building/animalshelter_':centroidsCategories[20],
                        '/building/apartment_':centroidsCategories[21],
                        '/building/auditorium_':centroidsCategories[22],
                        '/building/cityhall_':centroidsCategories[23],
                        '/building/conventioncenter_':centroidsCategories[24],
                        '/building/default_':centroidsCategories[25],
                        '/building/eventspace_':centroidsCategories[26],
                        '/building/factory_':centroidsCategories[27],
                        '/building/fair_':centroidsCategories[28],
                        '/building/funeralhome_':centroidsCategories[29],
                        '/building/government_':centroidsCategories[30],
                        '/building/gym_':centroidsCategories[31],
                        '/building/home_':centroidsCategories[32],
                        '/building/housingdevelopment_':centroidsCategories[33],
                        '/building/library_':centroidsCategories[34],
                        '/building/medical_':centroidsCategories[35],
                        '/building/militarybase_':centroidsCategories[36],
                        '/building/office_conferenceroom_':centroidsCategories[37],
                        '/building/office_coworkingspace_':centroidsCategories[38],
                        '/building/parking_':centroidsCategories[39],
                        '/building/postoffice_':centroidsCategories[40],
                        '/building/religious_':centroidsCategories[41],
                        '/building/school_':centroidsCategories[42],
                        '/building/votingbooth_':centroidsCategories[43],
                        '/education/academicbuilding_':centroidsCategories[44],
                        '/education/administrativebuilding_':centroidsCategories[45],
                        '/education/cafeteria_':centroidsCategories[46],
                        '/education/classroom_':centroidsCategories[47],
                        '/education/collegeacademicbuildings_communications_':centroidsCategories[48],
                        '/education/collegeacademicbuildings_engineering_':centroidsCategories[49],
                        '/education/collegeacademicbuildings_history_':centroidsCategories[50],
                        '/education/collegeacademicbuildings_math_':centroidsCategories[51],
                        '/education/collegeacademicbuildings_science_':centroidsCategories[52],
                        '/education/collegeacademicbuildings_technology_':centroidsCategories[53],
                        '/education/communitycollege_':centroidsCategories[54],
                        '/education/default_':centroidsCategories[55],
                        '/education/frathouse_':centroidsCategories[56],
                        '/education/lab_':centroidsCategories[57],
                        '/education/lawschool_':centroidsCategories[58],
                        '/education/other_':centroidsCategories[59],
                        '/education/quad_':centroidsCategories[60],
                        '/education/reccenter_':centroidsCategories[61],
                        '/education/residencehall_':centroidsCategories[62],
                        '/education/studentcenter_':centroidsCategories[63],
                        '/education/tradeschool_':centroidsCategories[64],
                        '/event/default_':centroidsCategories[65],
                        '/food/afghan_':centroidsCategories[66],
                        '/food/african_':centroidsCategories[67],
                        '/food/arepas_':centroidsCategories[68],
                        '/food/argentinian_':centroidsCategories[69],
                        '/food/asian_':centroidsCategories[70],
                        '/food/australian_':centroidsCategories[71],
                        '/food/austrian_':centroidsCategories[72],
                        '/food/bagels_':centroidsCategories[73],
                        '/food/bakery_':centroidsCategories[74],
                        '/food/bbqalt_':centroidsCategories[75],
                        '/food/breakfast_':centroidsCategories[76],
                        '/food/brewery_':centroidsCategories[77],
                        '/food/bubble_':centroidsCategories[78],
                        '/food/burger_':centroidsCategories[79],
                        '/food/burrito_':centroidsCategories[80],
                        '/food/cafe_':centroidsCategories[81],
                        '/food/cajun_':centroidsCategories[82],
                        '/food/caribbean_':centroidsCategories[83],
                        '/food/coffeeshop_':centroidsCategories[84],
                        '/food/creperie_':centroidsCategories[85],
                        '/food/cuban_':centroidsCategories[86],
                        '/food/cupcakes_':centroidsCategories[87],
                        '/food/default_':centroidsCategories[88],
                        '/food/deli_':centroidsCategories[89],
                        '/food/dessert_':centroidsCategories[90],
                        '/food/dimsum_':centroidsCategories[91],
                        '/food/diner_':centroidsCategories[92],
                        '/food/donuts_':centroidsCategories[93],
                        '/food/dumplings_':centroidsCategories[94],
                        '/food/ethiopian_':centroidsCategories[95],
                        '/food/falafel_':centroidsCategories[96],
                        '/food/fastfood_':centroidsCategories[97],
                        '/food/filipino_':centroidsCategories[98],
                        '/food/fishandchips_':centroidsCategories[99],
                        '/food/french_':centroidsCategories[100],
                        '/food/friedchicken_':centroidsCategories[101],
                        '/food/frozenyogurt_':centroidsCategories[102],
                        '/food/gastropub_':centroidsCategories[103],
                        '/food/german_':centroidsCategories[104],
                        '/food/glutenfree_':centroidsCategories[105],
                        '/food/greek_':centroidsCategories[106],
                        '/food/halal_':centroidsCategories[107],
                        '/food/hotdog_':centroidsCategories[108],
                        '/food/icecream_':centroidsCategories[109],
                        '/food/indian_':centroidsCategories[110],
                        '/food/indonesian_':centroidsCategories[111],
                        '/food/italian_':centroidsCategories[112],
                        '/food/japanese_':centroidsCategories[113],
                        '/food/juicebar_':centroidsCategories[114],
                        '/food/korean_':centroidsCategories[115],
                        '/food/latinamerican_':centroidsCategories[116],
                        '/food/macandcheese_':centroidsCategories[117],
                        '/food/malaysian_':centroidsCategories[118],
                        '/food/mediterranean_':centroidsCategories[119],
                        '/food/mexican_':centroidsCategories[120],
                        '/food/middleeastern_':centroidsCategories[121],
                        '/food/moleculargastronomy_':centroidsCategories[122],
                        '/food/moroccan_':centroidsCategories[123],
                        '/food/newamerican_':centroidsCategories[124],
                        '/food/paella_':centroidsCategories[125],
                        '/food/peruvian_':centroidsCategories[126],
                        '/food/pieshop_':centroidsCategories[127],
                        '/food/pizza_':centroidsCategories[128],
                        '/food/portuguese_':centroidsCategories[129],
                        '/food/ramen_':centroidsCategories[130],
                        '/food/russian_':centroidsCategories[131],
                        '/food/salad_':centroidsCategories[132],
                        '/food/scandinavian_':centroidsCategories[133],
                        '/food/seafood_':centroidsCategories[134],
                        '/food/snacks_':centroidsCategories[135],
                        '/food/soup_':centroidsCategories[136],
                        '/food/southern_':centroidsCategories[137],
                        '/food/spanish_':centroidsCategories[138],
                        '/food/steakhouse_':centroidsCategories[139],
                        '/food/streetfood_':centroidsCategories[140],
                        '/food/sushi_':centroidsCategories[141],
                        '/food/swiss_':centroidsCategories[142],
                        '/food/taco_':centroidsCategories[143],
                        '/food/tapas_':centroidsCategories[144],
                        '/food/tapas_':centroidsCategories[145],
                        '/food/tearoom_':centroidsCategories[146],
                        '/food/thai_':centroidsCategories[147],
                        '/food/turkish_':centroidsCategories[148],
                        '/food/vegetarian_':centroidsCategories[149],
                        '/food/vietnamese_':centroidsCategories[150],
                        '/food/winery_':centroidsCategories[151],
                        '/food/wings_':centroidsCategories[152],
                        '/nightlife/beergarden_':centroidsCategories[153],
                        '/nightlife/cocktails_':centroidsCategories[154],
                        '/nightlife/cocktails_':centroidsCategories[155],
                        '/nightlife/default_':centroidsCategories[156],
                        '/nightlife/divebar_':centroidsCategories[157],
                        '/nightlife/gaybar_':centroidsCategories[158],
                        '/nightlife/hookahbar_':centroidsCategories[159],
                        '/nightlife/karaoke_':centroidsCategories[160],
                        '/nightlife/nightclub_':centroidsCategories[161],
                        '/nightlife/pub_':centroidsCategories[162],
                        '/nightlife/pub_':centroidsCategories[163],
                        '/nightlife/sake_':centroidsCategories[164],
                        '/nightlife/secretbar_':centroidsCategories[165],
                        '/nightlife/sportsbar_':centroidsCategories[166],
                        '/nightlife/stripclub_':centroidsCategories[167],
                        '/nightlife/whiskey_':centroidsCategories[168],
                        '/parks_outdoors/baseballfield_':centroidsCategories[169],
                        '/parks_outdoors/basketballcourt_':centroidsCategories[170],
                        '/parks_outdoors/beach_':centroidsCategories[171],
                        '/parks_outdoors/botanicalgarden_':centroidsCategories[172],
                        '/parks_outdoors/bridge_':centroidsCategories[173],
                        '/parks_outdoors/campground_':centroidsCategories[174],
                        '/parks_outdoors/cemetery_':centroidsCategories[175],
                        '/parks_outdoors/default_':centroidsCategories[176],
                        '/parks_outdoors/dogrun_':centroidsCategories[177],
                        '/parks_outdoors/farm_':centroidsCategories[178],
                        '/parks_outdoors/field_':centroidsCategories[179],
                        '/parks_outdoors/garden_':centroidsCategories[180],
                        '/parks_outdoors/gardencenter_':centroidsCategories[181],
                        '/parks_outdoors/golfcourse_':centroidsCategories[182],
                        '/parks_outdoors/gun_':centroidsCategories[183],
                        '/parks_outdoors/harbor_':centroidsCategories[184],
                        '/parks_outdoors/hikingtrail_':centroidsCategories[185],
                        '/parks_outdoors/hotspring_':centroidsCategories[186],
                        '/parks_outdoors/lake_':centroidsCategories[187],
                        '/parks_outdoors/lighthouse_':centroidsCategories[188],
                        '/parks_outdoors/mountain_':centroidsCategories[189],
                        '/parks_outdoors/neighborhood_':centroidsCategories[190],
                        '/parks_outdoors/outdoors_':centroidsCategories[191],
                        '/parks_outdoors/park_':centroidsCategories[192],
                        '/parks_outdoors/playground_':centroidsCategories[193],
                        '/parks_outdoors/plaza_':centroidsCategories[194],
                        '/parks_outdoors/pool_':centroidsCategories[195],
                        '/parks_outdoors/river_':centroidsCategories[196],
                        '/parks_outdoors/sceniclookout_':centroidsCategories[197],
                        '/parks_outdoors/sculpture_':centroidsCategories[198],
                        '/parks_outdoors/skate_park_':centroidsCategories[199],
                        '/parks_outdoors/skatingrink_':centroidsCategories[200],
                        '/parks_outdoors/ski_apresskibar_':centroidsCategories[201],
                        '/parks_outdoors/ski_chairlift_':centroidsCategories[202],
                        '/parks_outdoors/ski_chalet_':centroidsCategories[203],
                        '/parks_outdoors/ski_lodge_':centroidsCategories[204],
                        '/parks_outdoors/ski_snowboard_':centroidsCategories[205],
                        '/parks_outdoors/ski_trail_':centroidsCategories[206],
                        '/parks_outdoors/stable_':centroidsCategories[207],
                        '/parks_outdoors/surfspot_':centroidsCategories[208],
                        '/parks_outdoors/vineyard_':centroidsCategories[209],
                        '/parks_outdoors/volleyballcourt_':centroidsCategories[210],
                        '/shops/airport_rentalcar_':centroidsCategories[211],
                        '/shops/antique_':centroidsCategories[212],
                        '/shops/apparel_':centroidsCategories[213],
                        '/shops/apparel_boutique_':centroidsCategories[214],
                        '/shops/artstore_':centroidsCategories[215],
                        '/shops/automotive_':centroidsCategories[216],
                        '/shops/beauty_cosmetic_':centroidsCategories[217],
                        '/shops/bikeshop_':centroidsCategories[218],
                        '/shops/bookstore_':centroidsCategories[219],
                        '/shops/bridal_':centroidsCategories[220],
                        '/shops/camerastore_':centroidsCategories[221],
                        '/shops/candystore_':centroidsCategories[222],
                        '/shops/carwash_':centroidsCategories[223],
                        '/shops/comic_':centroidsCategories[224],
                        '/shops/conveniencestore_':centroidsCategories[225],
                        '/shops/daycare_':centroidsCategories[226],
                        '/shops/default_':centroidsCategories[227],
                        '/shops/departmentstore_':centroidsCategories[228],
                        '/shops/design_':centroidsCategories[229],
                        '/shops/discountstore_':centroidsCategories[230],
                        '/shops/financial_':centroidsCategories[231],
                        '/shops/fleamarket_':centroidsCategories[232],
                        '/shops/flowershop_':centroidsCategories[233],
                        '/shops/food_butcher_':centroidsCategories[234],
                        '/shops/food_cheese_':centroidsCategories[235],
                        '/shops/food_farmersmarket_':centroidsCategories[236],
                        '/shops/food_fishmarket_':centroidsCategories[237],
                        '/shops/food_foodcourt_':centroidsCategories[238],
                        '/shops/food_gourmet_':centroidsCategories[239],
                        '/shops/food_grocery_':centroidsCategories[240],
                        '/shops/food_liquor_':centroidsCategories[241],
                        '/shops/food_wineshop_':centroidsCategories[242],
                        '/shops/foodanddrink_':centroidsCategories[243],
                        '/shops/furniture_':centroidsCategories[244],
                        '/shops/gamingcafe_':centroidsCategories[245],
                        '/shops/gas_':centroidsCategories[246],
                        '/shops/giftshop_':centroidsCategories[247],
                        '/shops/gym_martialarts_':centroidsCategories[248],
                        '/shops/gym_yogastudio_':centroidsCategories[249],
                        '/shops/hardware_':centroidsCategories[250],
                        '/shops/hobbyshop_':centroidsCategories[251],
                        '/shops/internetcafe_':centroidsCategories[252],
                        '/shops/jewelry_':centroidsCategories[253],
                        '/shops/laundry_':centroidsCategories[254],
                        '/shops/mall_':centroidsCategories[255],
                        '/shops/market_':centroidsCategories[256],
                        '/shops/mobilephoneshop_':centroidsCategories[257],
                        '/shops/motorcycle_':centroidsCategories[258],
                        '/shops/music_instruments_':centroidsCategories[259],
                        '/shops/nailsalon_':centroidsCategories[260],
                        '/shops/newsstand_':centroidsCategories[261],
                        '/shops/papergoods_':centroidsCategories[262],
                        '/shops/pet_store_':centroidsCategories[263],
                        '/shops/pharmacy_':centroidsCategories[264],
                        '/shops/photographylab_':centroidsCategories[265],
                        '/shops/realestate_':centroidsCategories[266],
                        '/shops/record_shop_':centroidsCategories[267],
                        '/shops/recycling_':centroidsCategories[268],
                        '/shops/salon_barber_':centroidsCategories[269],
                        '/shops/spa_':centroidsCategories[270],
                        '/shops/sports_outdoors_':centroidsCategories[271],
                        '/shops/storage_':centroidsCategories[272],
                        '/shops/tanning_salon_':centroidsCategories[273],
                        '/shops/tattoos_':centroidsCategories[274],
                        '/shops/technology_':centroidsCategories[275],
                        '/shops/tobacco_':centroidsCategories[276],
                        '/shops/toys_':centroidsCategories[277],
                        '/shops/video_':centroidsCategories[278],
                        '/shops/videogames_':centroidsCategories[279],
                        '/travel/airport_':centroidsCategories[280],
                        '/travel/bedandbreakfast_':centroidsCategories[281],
                        '/travel/boat_':centroidsCategories[282],
                        '/travel/busstation_':centroidsCategories[283],
                        '/travel/default_':centroidsCategories[284],
                        '/travel/embassy_':centroidsCategories[285],
                        '/travel/ferry_pier_':centroidsCategories[286],
                        '/travel/highway_':centroidsCategories[287],
                        '/travel/hostel_':centroidsCategories[288],
                        '/travel/hotel_':centroidsCategories[289],
                        '/travel/lightrail_':centroidsCategories[290],
                        '/travel/movingtarget_':centroidsCategories[291],
                        '/travel/resort_':centroidsCategories[292],
                        '/travel/restarea_':centroidsCategories[293],
                        '/travel/subway_':centroidsCategories[294],
                        '/travel/taxi_':centroidsCategories[295],
                        '/travel/touristinformation_':centroidsCategories[296],
                        '/travel/trainstation_':centroidsCategories[297],
                        '/travel/travelagency_':centroidsCategories[298],
                        '/food/tibetan_':centroidsCategories[299],
                        'Unknown':centroidsCategories[300]}
    cats = sorted(categoriesDict.items(), key=lambda x: x[1])
    for key, value in cats:
        if (key == 0): cats[value] = 'NA'
    return cats

#--------------Temporal Trainer------------------------------------
def temporalTrainer (trainingDataset):

    # Finding Distinct User IDs of Training Dataset
    userIDs2 = []
    for line in trainingDataset:
        userIDs2.append(line[0])
    userIDs2 = np.unique(userIDs2)
    userIDsAmount = 0
    for userz in userIDs2:
        userIDsAmount+=1
    temporalArray = np.zeros((userIDsAmount, 1680))
    i = 0
    # Creating Temporal Array
    for user in userIDs2:
        for z in trainingDataset:
            if (trainingDataset[0] == user):
                for y in categoriez:
                    if (trainingDataset[1] == categoriez[0]):
                        Day = datetime.strptime(trainingDataset[2], '%a %b %d %H:%M:%S').strftime('%a')
                        Day = (Day - 1)*24
                        Hour = datetime.strptime(trainingDataset[2], '%a %b %d %H:%M:%S').strftime('%H')
                        max = (10)*(Hour + Day)
                        min = max - 9

                        if (categoriez[1] == "Arts & Entertainment"):
                            temporalArray[i][min] += 1
                        elif (categoriez[1] == "College & University"):
                            temporalArray[i][min+1] += 1
                        elif (categoriez[1] == "Event"):
                            temporalArray[i][min+2] += 1
                        elif (categoriez[1] == "Food"):
                            temporalArray[i][min+3] += 1
                        elif (categoriez[1] == "Nightlife Spot"):
                            temporalArray[i][min+4] += 1
                        elif (categoriez[1] == "Outdoors & Recreation"):
                            temporalArray[i][min+5] += 1
                        elif (categoriez[1] == "Professional & Other Places"):
                            temporalArray[i][min+6] += 1
                        elif (categoriez[1] == "Residences"):
                            temporalArray[i][min+7] += 1
                        elif (categoriez[1] == "Shop & Service"):
                            temporalArray[i][min+8] += 1
                        elif (categoriez[1] == "Travel & Transport"):
                            temporalArray[i][min+9] += 1
                        break
                        '''
                        Array extraction equation

                        1) Get Time and Day
                        2) Day = (Day - 1)*24
                        3) MaxIndex = (10)*[Hour+Day]
                        4) MinIndex = MaxIndex - 9

                        '''
                return 1
        i+=1
    return 0

#--------------Training Optimization (Frequency of check-ins,etc.)-
def training (CC,sum2):
    initializer = 0
    sum = 0
    row = 0
    numofCats = 0
    for i in CC:
        for ii in i:
            if (initializer > 1):
                sum += ii
                if (ii != 0):
                    numofCats += 1
            initializer += 1
        sfreq = float(sum) / float(sum2)

        initializer = 0
        # Equation 4, 5, 6 in the MUAPLUSTC Paper
        Hur = 0
        for ii in i:
            if (initializer > 1):
                if (ii != 0):
                    probability = float (ii) / float (sum)
                    Hur += probability * mt.log(probability,2)
            initializer += 1
        Hur *= -1
        HMax = mt.log(numofCats,2)
        try:
            ratioPB = 1 - (float(Hur)/float(HMax))
            if (sum == 0 or sfreq < 0.01 or ratioPB < 0.4):
                try:
                    CC = np.delete(CC, (row), axis=0)  # np.delete(array, (row index), axis = 0)
                except IndexError:
                    pass

        except ZeroDivisionError:
            pass

        row += 1
        initializer = 0

# Main Category Parameters
'''

        Main Category Variables
        XA --> Actual Values | XP --> Predicted Values
        AER -> Arts&Entertainment
        CU -> College & University
        FOO -> Food
        NS -> Nightlife Spot
        ODR -> Outdoor Recreation
        PO -> Professional & Other Places
        RES -> Residence
        SHOSER -> Shops & Services
        TT -> Travel and Transport


'''

# XMeans Clustering Class
class XMeans:
    """
    x-means
    """

    def __init__(self, k_init=2, **k_means_args):
        """
        k_init : The initial number of clusters applied to KMeans()
        """
        self.k_init = k_init
        self.k_means_args = k_means_args

    def fit(self, X):
        """
        x-means
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        self.__clusters = []

        clusters = self.Cluster.build(X, KMeans(self.k_init, **self.k_means_args).fit(X))
        self.__recursively_split(clusters)

        self.labels_ = np.empty(X.shape[0], dtype=np.intp)
        for i, c in enumerate(self.__clusters):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])

        return self


    def __recursively_split(self, clusters):
        """
      clusters
        clusters : list-like object, which contains instances of 'XMeans.Cluster'
        """
        for cluster in clusters:
            if cluster.size <= 3:
                self.__clusters.append(cluster)
                continue

            k_means = KMeans(2, **self.k_means_args).fit(cluster.data)
            c1, c2 = self.Cluster.build(cluster.data, k_means, cluster.index)

            beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            alpha = 0.5 / stats.norm.cdf(beta)
            bic = -2 * (
            cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df * np.log(
                cluster.size)

            if bic < cluster.bic():
                self.__recursively_split([c1, c2])
            else:
                self.__clusters.append(cluster)

    class Cluster:
        """
        k-means
        """

        @classmethod
        def build(cls, X, k_means, index=None):
            if index == None:
                index = np.array(range(0, X.shape[0]))
            labels = range(0, k_means.get_params()["n_clusters"])

            return tuple(cls(X, index, k_means, label) for label in labels)

        # index: X
        def __init__(self, X, index, k_means, label):
            self.data = X[k_means.labels_ == label]
            self.index = index[k_means.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            self.center = k_means.cluster_centers_[label]
            self.cov = np.cov(self.data.T)

        def log_likelihood(self):
            return sum(stats.multivariate_normal.logpdf(x, self.center, self.cov) for x in self.data)

        def bic(self):
            return -2 * self.log_likelihood() + self.df * np.log(self.size)


if __name__ == "__main__":
    # Subcategories List
    subcategories = ['/arts_entertainment/aquarium_',
                        '/arts_entertainment/arcade_',
                        '/arts_entertainment/artgallery_',
                        '/arts_entertainment/billiards_',
                        '/arts_entertainment/bowling_',
                        '/arts_entertainment/casino_',
                        '/arts_entertainment/comedyclub_',
                        '/arts_entertainment/default_',
                        '/arts_entertainment/historicsite_',
                        '/arts_entertainment/movietheater_',
                        '/arts_entertainment/museum_',
                        '/arts_entertainment/musicvenue_',
                        '/arts_entertainment/performingarts_',
                        '/arts_entertainment/racetrack_',
                        '/arts_entertainment/stadium_',
                        '/arts_entertainment/themepark_',
                        '/arts_entertainment/waterpark_',
                        '/arts_entertainment/zoo_',
                        '/building/animalshelter_',
                        '/building/apartment_',
                        '/building/auditorium_',
                        '/building/cityhall_',
                        '/building/conventioncenter_',
                        '/building/default_',
                        '/building/eventspace_',
                        '/building/factory_',
                        '/building/fair_',
                        '/building/funeralhome_',
                        '/building/government_',
                        '/building/gym_',
                        '/building/home_',
                        '/building/housingdevelopment_',
                        '/building/library_',
                        '/building/medical_',
                        '/building/militarybase_',
                        '/building/office_conferenceroom_',
                        '/building/office_coworkingspace_',
                        '/building/parking_',
                        '/building/postoffice_',
                        '/building/religious_',
                        '/building/school_',
                        '/building/votingbooth_',
                        '/education/academicbuilding_',
                        '/education/administrativebuilding_',
                        '/education/cafeteria_',
                        '/education/classroom_',
                        '/education/collegeacademicbuildings_communications_',
                        '/education/collegeacademicbuildings_engineering_',
                        '/education/collegeacademicbuildings_history_',
                        '/education/collegeacademicbuildings_math_',
                        '/education/collegeacademicbuildings_science_',
                        '/education/collegeacademicbuildings_technology_',
                        '/education/communitycollege_',
                        '/education/default_',
                        '/education/frathouse_',
                        '/education/lab_',
                        '/education/lawschool_',
                        '/education/other_',
                        '/education/quad_',
                        '/education/reccenter_',
                        '/education/residencehall_',
                        '/education/studentcenter_',
                        '/education/tradeschool_',
                        '/event/default_',
                        '/food/afghan_',
                        '/food/african_',
                        '/food/arepas_',
                        '/food/argentinian_',
                        '/food/asian_',
                        '/food/australian_',
                        '/food/austrian_',
                        '/food/bagels_',
                        '/food/bakery_',
                        '/food/bbqalt_',
                        '/food/breakfast_',
                        '/food/brewery_',
                        '/food/bubble_',
                        '/food/burger_',
                        '/food/burrito_',
                        '/food/cafe_',
                        '/food/cajun_',
                        '/food/caribbean_',
                        '/food/coffeeshop_',
                        '/food/creperie_',
                        '/food/cuban_',
                        '/food/cupcakes_',
                        '/food/default_',
                        '/food/deli_',
                        '/food/dessert_',
                        '/food/dimsum_',
                        '/food/diner_',
                        '/food/donuts_',
                        '/food/dumplings_',
                        '/food/ethiopian_',
                        '/food/falafel_',
                        '/food/fastfood_',
                        '/food/filipino_',
                        '/food/fishandchips_',
                        '/food/french_',
                        '/food/friedchicken_',
                        '/food/frozenyogurt_',
                        '/food/gastropub_',
                        '/food/german_',
                        '/food/glutenfree_',
                        '/food/greek_',
                        '/food/halal_',
                        '/food/hotdog_',
                        '/food/icecream_',
                        '/food/indian_',
                        '/food/indonesian_',
                        '/food/italian_',
                        '/food/japanese_',
                        '/food/juicebar_',
                        '/food/korean_',
                        '/food/latinamerican_',
                        '/food/macandcheese_',
                        '/food/malaysian_',
                        '/food/mediterranean_',
                        '/food/mexican_',
                        '/food/middleeastern_',
                        '/food/moleculargastronomy_',
                        '/food/moroccan_',
                        '/food/newamerican_',
                        '/food/paella_',
                        '/food/peruvian_',
                        '/food/pieshop_',
                        '/food/pizza_',
                        '/food/portuguese_',
                        '/food/ramen_',
                        '/food/russian_',
                        '/food/salad_',
                        '/food/scandinavian_',
                        '/food/seafood_',
                        '/food/snacks_',
                        '/food/soup_',
                        '/food/southern_',
                        '/food/spanish_',
                        '/food/steakhouse_',
                        '/food/streetfood_',
                        '/food/sushi_',
                        '/food/swiss_',
                        '/food/taco_',
                        '/food/tapas_',
                        '/food/tapas_',
                        '/food/tearoom_',
                        '/food/thai_',
                        '/food/turkish_',
                        '/food/vegetarian_',
                        '/food/vietnamese_',
                        '/food/winery_',
                        '/food/wings_',
                        '/nightlife/beergarden_',
                        '/nightlife/cocktails_',
                        '/nightlife/cocktails_',
                        '/nightlife/default_',
                        '/nightlife/divebar_',
                        '/nightlife/gaybar_',
                        '/nightlife/hookahbar_',
                        '/nightlife/karaoke_',
                        '/nightlife/nightclub_',
                        '/nightlife/pub_',
                        '/nightlife/pub_',
                        '/nightlife/sake_',
                        '/nightlife/secretbar_',
                        '/nightlife/sportsbar_',
                        '/nightlife/stripclub_',
                        '/nightlife/whiskey_',
                        '/parks_outdoors/baseballfield_',
                        '/parks_outdoors/basketballcourt_',
                        '/parks_outdoors/beach_',
                        '/parks_outdoors/botanicalgarden_',
                        '/parks_outdoors/bridge_',
                        '/parks_outdoors/campground_',
                        '/parks_outdoors/cemetery_',
                        '/parks_outdoors/default_',
                        '/parks_outdoors/dogrun_',
                        '/parks_outdoors/farm_',
                        '/parks_outdoors/field_',
                        '/parks_outdoors/garden_',
                        '/parks_outdoors/gardencenter_',
                        '/parks_outdoors/golfcourse_',
                        '/parks_outdoors/gun_',
                        '/parks_outdoors/harbor_',
                        '/parks_outdoors/hikingtrail_',
                        '/parks_outdoors/hotspring_',
                        '/parks_outdoors/lake_',
                        '/parks_outdoors/lighthouse_',
                        '/parks_outdoors/mountain_',
                        '/parks_outdoors/neighborhood_',
                        '/parks_outdoors/outdoors_',
                        '/parks_outdoors/park_',
                        '/parks_outdoors/playground_',
                        '/parks_outdoors/plaza_',
                        '/parks_outdoors/pool_',
                        '/parks_outdoors/river_',
                        '/parks_outdoors/sceniclookout_',
                        '/parks_outdoors/sculpture_',
                        '/parks_outdoors/skate_park_',
                        '/parks_outdoors/skatingrink_',
                        '/parks_outdoors/ski_apresskibar_',
                        '/parks_outdoors/ski_chairlift_',
                        '/parks_outdoors/ski_chalet_',
                        '/parks_outdoors/ski_lodge_',
                        '/parks_outdoors/ski_snowboard_',
                        '/parks_outdoors/ski_trail_',
                        '/parks_outdoors/stable_',
                        '/parks_outdoors/surfspot_',
                        '/parks_outdoors/vineyard_',
                        '/parks_outdoors/volleyballcourt_',
                        '/shops/airport_rentalcar_',
                        '/shops/antique_',
                        '/shops/apparel_',
                        '/shops/apparel_boutique_',
                        '/shops/artstore_',
                        '/shops/automotive_',
                        '/shops/beauty_cosmetic_',
                        '/shops/bikeshop_',
                        '/shops/bookstore_',
                        '/shops/bridal_',
                        '/shops/camerastore_',
                        '/shops/candystore_',
                        '/shops/carwash_',
                        '/shops/comic_',
                        '/shops/conveniencestore_',
                        '/shops/daycare_',
                        '/shops/default_',
                        '/shops/departmentstore_',
                        '/shops/design_',
                        '/shops/discountstore_',
                        '/shops/financial_',
                        '/shops/fleamarket_',
                        '/shops/flowershop_',
                        '/shops/food_butcher_',
                        '/shops/food_cheese_',
                        '/shops/food_farmersmarket_',
                        '/shops/food_fishmarket_',
                        '/shops/food_foodcourt_',
                        '/shops/food_gourmet_',
                        '/shops/food_grocery_',
                        '/shops/food_liquor_',
                        '/shops/food_wineshop_',
                        '/shops/foodanddrink_',
                        '/shops/furniture_',
                        '/shops/gamingcafe_',
                        '/shops/gas_',
                        '/shops/giftshop_',
                        '/shops/gym_martialarts_',
                        '/shops/gym_yogastudio_',
                        '/shops/hardware_',
                        '/shops/hobbyshop_',
                        '/shops/internetcafe_',
                        '/shops/jewelry_',
                        '/shops/laundry_',
                        '/shops/mall_',
                        '/shops/market_',
                        '/shops/mobilephoneshop_',
                        '/shops/motorcycle_',
                        '/shops/music_instruments_',
                        '/shops/nailsalon_',
                        '/shops/newsstand_',
                        '/shops/papergoods_',
                        '/shops/pet_store_',
                        '/shops/pharmacy_',
                        '/shops/photographylab_',
                        '/shops/realestate_',
                        '/shops/record_shop_',
                        '/shops/recycling_',
                        '/shops/salon_barber_',
                        '/shops/spa_',
                        '/shops/sports_outdoors_',
                        '/shops/storage_',
                        '/shops/tanning_salon_',
                        '/shops/tattoos_',
                        '/shops/technology_',
                        '/shops/tobacco_',
                        '/shops/toys_',
                        '/shops/video_',
                        '/shops/videogames_',
                        '/travel/airport_',
                        '/travel/bedandbreakfast_',
                        '/travel/boat_',
                        '/travel/busstation_',
                        '/travel/default_',
                        '/travel/embassy_',
                        '/travel/ferry_pier_',
                        '/travel/highway_',
                        '/travel/hostel_',
                        '/travel/hotel_',
                        '/travel/lightrail_',
                        '/travel/movingtarget_',
                        '/travel/resort_',
                        '/travel/restarea_',
                        '/travel/subway_',
                        '/travel/taxi_',
                        '/travel/touristinformation_',
                        '/travel/trainstation_',
                        '/travel/travelagency_',
                        '/food/tibetan_',
                        'Unknown']
    # Results Parameters
    accuracyusers = 0
    AERA = 0
    AERCP = 0
    AERP = 0
    CUA = 0
    CUP = 0
    CUCP = 0
    FOOA = 0
    FOOP = 0
    FOOCP = 0
    NSA = 0
    NSP = 0
    NSCP = 0
    ODRA = 0
    ODRP = 0
    ODRCP = 0
    POA = 0
    POP = 0
    POCP = 0
    RESA = 0
    RESP = 0
    RESCP = 0
    SHOSERA = 0
    SHOSERP = 0
    SHOSERCP = 0
    TTA = 0
    TTP = 0
    TTCP = 0
    accuracy = 0

    datasetCheckins = 0


    # Temporal Training Array
    tempos = []
    # Spatial Training Array
    geo = []
    # Note: A Dataset with at least 10 months of check-ins is needed for better results
    # Note2: If the dataset hass less than 10 months, try training the model using the first 80-90 percent of your data
    # Parameters needed in the dataset: Unique User IDs (encrypted user IDs are okay) | Venue IDs (Foursquare) | LO and LA of check-ins | Time of check-ins
    # Training Dataset (First 8-9 months of the dataset)
    with open('C:\Users\Alireza\Desktop\Datasets\New York\Training.txt') as inputfile:
        for line in inputfile:
            # User IDS
            tempos.append((line.split('\t'))[0])
            geo.append((line.split('\t'))[0])
            # Venue IDS
            geo.append((line.split('\t'))[1])
            tempos.append((line.split('\t'))[1])
            # Spatial LO/LA
            geo.append((line.split('\t'))[4])
            geo.append((line.split('\t'))[5])
            # Temporal (Time)
            tempos.append((line.split('\t'))[7])
    # For Spatial
    A = np.array(geo)
    B = np.reshape(A, (-1, 4))
    # For Temporal
    C = np.array (tempos)
    C = np.reshape (C, (-1,3))

    # Training Temporal
    #temporal = temporalTrainer(C)

    # Testing Dataset
    with open('C:\Users\Alireza\Desktop\Datasets\New York\Testing.txt') as inputfile2:
        # Testing Spatial
        testGeo = []
        # Testing temporal
        testTempos = []
        # User IDs
        users = []
        for line in inputfile2:
            testGeo.append((line.split('\t'))[0])
            users.append((line.split('\t'))[0])
            testGeo.append((line.split('\t'))[1])
            testGeo.append((line.split('\t'))[4])
            testGeo.append((line.split('\t'))[5])

    testGeo = np.array(testGeo)
    testGeo = np.reshape(testGeo, (-1, 4))


    # Unique user IDs (used for testing)
    userIDs = np.unique(users)


    # Appending Venue IDs and LO/LA for each user
    userNumber = 0
    totalUsers = userIDs.size
    tests = 0
    passed = 0
    for user in userIDs:
        venueIDsAndLL = []
        for z in B:
            if (z[0] == user):
                venueIDsAndLL.append(z[1])
                venueIDsAndLL.append(z[2])
                venueIDsAndLL.append(z[3])

        venueIDsAndLL = np.array(venueIDsAndLL)
        venueIDsAndLL = np.reshape(venueIDsAndLL, (-1, 3))


        longtituteLangtitute = []

        for z in venueIDsAndLL:
            longtituteLangtitute.append(z[1])
            longtituteLangtitute.append(z[2])



        longtituteLangtitute = np.array(longtituteLangtitute)
        longtituteLangtitute = np.array(longtituteLangtitute).tolist()

        longtituteLangtituteFloat = []

        for zz in longtituteLangtitute:
            longtituteLangtituteFloat.append(float(zz))


        longtituteLangtituteFloat = np.array(longtituteLangtituteFloat)
        longtituteLangtituteFloat = np.reshape(longtituteLangtituteFloat, (-1, 2))
        longtituteLangtituteFloat = np.array(longtituteLangtituteFloat)
        longtituteLangtituteFloat = np.array(longtituteLangtituteFloat).tolist()
        longtituteLangtituteFloat = longtituteLangtituteFloat * 1  # Performance differs for values between 1 to 100, try 10 or 100 if performance is low

        # Clustering Process
        try :
            x_means = XMeans().fit(np.c_[longtituteLangtituteFloat])
        except ValueError:
            continue
        # Amount of clusters created (PFRs of the user)
        centroidsAmount = x_means.cluster_centers_.size/2
        #----Category-Centroids
        centroidsCategories = np.zeros((centroidsAmount, 301))

        i = 0
        for z in x_means.cluster_centers_:
            centroidsCategories[i][0] = z[0]
            centroidsCategories[i][1] = z[1]
            i += 1

        # Adding the checked-in categories to the array of PFRs
        sum1 = 0
        for z in venueIDsAndLL:
            for y in categoriez:
                if (z[0] == y[0]):
                    nc = nearest_cluster(z[1],z[2])
                    try:
                        m = subcategories.index(y[1])
                        m+=2
                        centroidsCategories[nc][m] += 1
                        sum1 += 1
                    except ValueError:
                        break
                    break

        # --------------------------------------------------------
        # ----------------------Testing---------------------------
        # --------------------------------------------------------

        training(centroidsCategories,sum1)

        venueIDsAndLL2 = []
        for z in testGeo:
            if (z[0] == user):
                venueIDsAndLL2.append(z[1])
                venueIDsAndLL2.append(z[2])
                venueIDsAndLL2.append(z[3])

        venueIDsAndLL2 = np.array(venueIDsAndLL2)
        venueIDsAndLL2 = np.reshape(venueIDsAndLL2, (-1, 3))

        max = 0
        userpassed = 0
        userTests = 0
        at3 = 0

        for z in venueIDsAndLL2:
            for y in categoriez:

                if (z[0] == y[0]):
                    category = y[1]
                    break

            # finding the nearest cluster index
            nc = nearest_cluster(z[1], z[2])
            # sorting the category check-ins by highest to lowest
            CC = centroidsCategories[nc]
            # Used for measuring accuracies
            completed = 0
            # Amount of cells of the training array | Index 0 to 1 LA and LO of centroids | Index 2:.. Subcategories
            i = 295
            while (completed < 1): # Completed < n --> Accuracy at n
                HC = categories(CC)
                # Checking whether the predicted value equals the actual value of the testing portion of the dataset
                if (category == HC[i][0] or optimized(category,HC[i][0]) > 0.5):
                    # Actual Values
                    if (category.find("/arts_entertainment") != -1):
                        AERA += 1
                    if (category.find("/education/") != -1):
                        CUA += 1
                    if (category.find("/food/") != -1):
                        FOOA += 1
                    if (category.find("/nightlife/") != -1):
                        NSA += 1
                    if (category.find("/parks_outdoors/") != -1):
                        ODRA += 1
                    if (category.find("/building/") != -1):
                        POA += 1
                    if ((category.find("/building/home_") != -1) or (category.find("/building/apartment_") != -1) or (category.find("/building/housingdevelopment_") != -1)):
                        RESA += 1
                    if (category.find("/shops/") != -1):
                        SHOSERA += 1
                    if (category.find("/travel/") != -1):
                        TTA += 1
                    # True Positives
                    if (HC[i][0].find("/arts_entertainment") != -1 and category.find("/arts_entertainment") != -1):
                        AERCP += 1
                    elif (HC[i][0].find("/education/") != -1 and category.find("/education/") != -1):
                        CUCP += 1
                    elif (HC[i][0].find("/food/") != -1 and category.find("/food/") != -1):
                        FOOCP += 1
                    elif (HC[i][0].find("/nightlife/") != -1 and category.find("/nightlife/") != -1):
                        NSCP += 1
                    elif (HC[i][0].find("/parks_outdoors/") != -1 and category.find("/parks_outdoors/") != -1):
                        ODRCP += 1
                    elif (HC[i][0].find("/building/") != -1 and category.find("/building/") != -1):
                        POCP += 1
                    elif ((((HC[i][0].find("/building/home") != -1) or (HC[i][0].find("/building/apartment") != -1) or (HC[i][0].find("/building/housingdevelopment") != -1))) and (((category.find("/building/home") != -1) or (category.find("/building/apartment") != -1) or (category.find("/building/housingdevelopment") != -1)))):
                        RESCP += 1
                    elif (HC[i][0].find("/shops/") != -1 and category.find("/shops/") != -1):
                        SHOSERCP += 1
                    elif (HC[i][0].find("/travel/") != -1 and category.find("/travel/") != -1):
                        TTCP += 1
                    if (HC[i][0].find("/arts_entertainment") != -1):
                        AERP += 1
                    if (HC[i][0].find("/education/") != -1):
                        CUP += 1
                    if (HC[i][0].find("/food/") != -1):
                        FOOP += 1
                    if (HC[i][0].find("/nightlife/") != -1):
                        NSP += 1
                    if (HC[i][0].find("/parks_outdoors/") != -1):
                        ODRP += 1
                    if (HC[i][0].find("/building/")):
                        POP += 1
                    if (HC[i][0].find("/building/home_") != -1 or HC[i][0].find("/building/apartment_") != -1 or HC[i][0].find("/building/housingdevelopment_") != -1):
                        RESP += 1
                    if (HC[i][0].find("/shops/") != -1):
                        SHOSERP += 1
                    if (HC[i][0].find("/travel/") != -1):
                        TTP += 1
                    passed +=1
                    userpassed+=1
                    userTests+=1
                    tests +=1
                    break
                else:
                    i -= 1
                    completed+=1
                # Accuracy @: {at<0 @1, at<5 @5, at<10 @10}: Completed = 1, 5, 10
                if (completed ==1):
                    if (HC[i][0].find("/arts_entertainment") != -1):
                        AERP += 1
                    if (HC[i][0].find("/education/") != -1):
                        CUP += 1
                    if (HC[i][0].find("/food/") != -1):
                        FOOP += 1
                    if (HC[i][0].find("/nightlife/") != -1):
                        NSP += 1
                    if (HC[i][0].find("/parks_outdoors/") != -1):
                        ODRP += 1
                    if (HC[i][0].find("/building/")):
                        POP += 1
                    if (HC[i][0].find("/building/home_") != -1 or HC[i][0].find("/building/apartment_") != -1 or HC[i][0].find("/building/housingdevelopment_") != -1):
                        RESP += 1
                    if (HC[i][0].find("/shops/") != -1):
                        SHOSERP += 1
                    if (HC[i][0].find("/travel/") != -1):
                        TTP += 1
                    if (category.find("/arts_entertainment") != -1):
                        AERA += 1
                    if (category.find("/education/") != -1):
                        CUA += 1
                    if (category.find("/food/") != -1):
                        FOOA += 1
                    if (category.find("/nightlife/") != -1):
                        NSA += 1
                    if (category.find("/parks_outdoors/") != -1):
                        ODRA += 1
                    if (category.find("/building/") != -1):
                        POA += 1
                    if (category.find("/building/home_") != -1 or category.find("/building/apartment_") != -1 or category.find("/building/housingdevelopment_") != -1):
                        RESA += 1
                    if (category.find("/shops/") != -1):
                        SHOSERA += 1
                    if (category.find("/travel/") != -1):
                        TTA += 1
                    tests+=1
                    userTests+=1
                    break
        userNumber += 1
        print 'User %d out of Users %d' % (userNumber, totalUsers)
        #results.write("User %d results\t" % userNumber)
        try:
            accuracy = (float(userpassed)/float(userTests))*100
            accuracyf = (float(passed) / float(tests)) * 100
            accuracyusers = (accuracyusers + accuracy)/userNumber
        except (ZeroDivisionError):
            accuracyf = (float(passed) / float(tests)) * 100
            accuracy = 0
        print ("Final Accuracy = %%%lf (whole system)" % accuracyf)



        try:
            # Precision = Correct Predictions / All Predictions
            AEPrecision = (float(AERCP) / float(AERP))*100
            CUPrecision = (float(CUCP) / float(CUP))*100
            FPrecision = (float(FOOCP) / float(FOOP))*100
            NPrecision = (float(NSCP) / float(NSP))*100
            ORPrecision = (float(ODRCP) / float(ODRP))*100
            POPrecision = (float(POCP) / float(POP))*100
            RPrecision = (float(RESCP) / float(RESP))*100
            SPrecision = (float(SHOSERCP) / float(SHOSERP))*100
            TTPrecision = (float(TTCP) / float(TTP))*100
            # Recall = Correct Predictions / Actual Values
            AERecall = (float(AERCP) / float(AERA)) * 100
            CURecall = (float(CUCP) / float(CUA)) * 100
            FRecall = (float(FOOCP) / float(FOOA)) * 100
            NSRecall = (float(NSCP) / float(NSA)) * 100
            ORRecall = (float(ODRCP) / float(ODRA)) * 100
            POPREcall = (float(POCP) / float(POA)) * 100
            RRecall = (float(RESCP) / float(RESA)) * 100
            SRecall = (float(SHOSERCP) / float(SHOSERA)) * 100
            TTRecall = (float(TTCP) / float(TTA)) * 100
        except (ZeroDivisionError):
            continue

# Uncomment for checking the results
'''
accuracyf = (float(passed) / float(tests)) * 100
print ("Final Accuracy = %%%lf (whole system)\n\n" % accuracyf)
print("Arts and Entertainment Precision = %%%lf\n" % AEPrecision)
print("College & University Precision = %%%lf\n" % CUPrecision)
print ("Food Precision = %%%lf \n" % FPrecision)
print ("Nightlife Spot Precision = %%%lf \n" %  NPrecision)
print("Outdoor Recreation Precision = %%%lf \n" %  ORPrecision)
print ("Professional and Other Places Precision = %%%lf\n" % POPrecision)
print ("Residence Precision = %%%lf\n" % RPrecision)
print ("Shops and Service Precision = %%%lf\n" % SPrecision)
print ("Travel and Transport Precision = %%%lf\n\n\n" % TTPrecision)

print("Arts and Entertainment Recall = %%%lf\n" % AERecall)
print("College & University Recall = %%%lf\n" % CURecall)
print ("Food Recall = %%%lf \n" % FRecall)
print ("Nightlife Spot Recall = %%%lf \n" % NSRecall)
print("Outdoor Recreation Recall = %%%lf \n" % ORRecall)
print ("Professional and Other Places Recall = %%%lf\n" % POPREcall)
print ("Residence Recall = %%%lf\n" % RRecall)
print ("Shops and Service Recall = %%%lf\n" % SRecall)
print ("Travel and Transport Recall = %%%lf\n" % TTRecall)
'''