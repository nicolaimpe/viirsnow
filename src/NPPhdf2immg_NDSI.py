#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# @(#)NPPhdf2immg.py version 1.1 07/04/13 (C) METEO-FRANCE
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Modification pour calculer le NDSI et la fraction neigeuse
# Marie Dumont, Juillet 2015
#++
# NOM
#
# SOMMAIRE
#    ...
# PACKAGE
#    ArchiPEL_npp
# SYNTAXE
#     options FILE_SRC FILE_PRD
# DESCRIPTION
#    NPPnetcdf2img est le script de fabrication des produits monocanaux
#    issus de la reception locale de NPP(VIIRS) ; il est lance par
#     l application imgVIIRS.
# OPTIONS
#    -S srcid
#    Identificateur de la source.
#    Cette cle est obligatoire.
#
#    -s satimg
#    Identificateur du satellite imageur
#    Cette cle est obligatoire.
#
#    -d yyyymddd
#    Date de reference de la source
#    Cette cle est obligatoire.
#
#    -h hhmn
#    Heure de reference de la source
#    Cette cle est obligatoire.
#
#    -n nnnnn
#    Indice de reference (slot) de la source
#    Cette cle est obligatoire.
# ENVIRONNEMENT
#    Environnement ArchiPEL 2
# FICHIERS
#    fichier de definition
#      <prdid>.def  fichier de definition du produit prdid
# NOTES
#
# VOIR AUSSI
# AUTEUR(S)
#    S Guevel
# DATE CREATION
#
# DERNIERE MODIFICATION
#    07/04/13
# VERSION.RELEASE
#    1.1
#--
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# environnement
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# necessite le module pyresample, entre autres modules python

import numpy as np
import numpy.random
import os
import logging
import argparse
import pyresample
import re
import time
import sys
import math
import h5py



def if_else(condition, trueVal, falseVal):
    if condition:
        return trueVal
    else:
        return falseVal

def getDynamicFunctionFromData(min, max):
    import logging
    logging.debug("getDynamicFunctionFromData")
    if max==min:
        return lambda x : x
    else:
        return lambda x : (x-min)/(max-min)*255


def getDynamicFunctionFromFile(file):
    # fichier est une suite de lignes du type phy cn (ce qui suit est ignore)
    # interpolation realisee entre chaque palier par fonction np.interp
    # lignes commencant par # sont des commentaires
    logging.debug("getDynamicFunctionFromFile")

    # opening file
    try:
        f = open(file)
    except IOError:
        logging.error("Cannot open file %s" % file)
        return None

    phys=[]
    cns=[]
    for line in f:
        if line.startswith("#"):
                continue
    m = re.search("(?P<phys>\d*?[\.]\d*)\s*(?P<cn>\d+)", line)
        phy, cn = m.groups()
        phys.append(phy)
        cns.append(cn)

    return lambda x: np.interp(x, phys, cns)

# Calcul du min des parametres phys d'un fichier de dyn
#  [ passe en colonne une du fichier ]
def getMinFromFileDyn(file):

    logging.debug("getMinFromFileDyn")
    try:
        f = open(file)
    except IOError:
        logging.error("Cannot open file %s" % file)
        return None

    phys=[]
    cns=[]
    for line in f:
        if line.startswith("#"):
                continue
        m = re.search("(?P<phys>\d*?[\.]\d*)\s*(?P<cn>\d+)", line)
    phy, cn = m.groups()
        phys.append(phy)
        cns.append(cn)
    physA=np.array([phys]).astype(float)
    minphy=physA.min()
    return minphy

# Calcul du max des parametres phys d'un fichier de dyn
#  [ passe en colonne une du fichier ]
def getMaxFromFileDyn(file):

    logging.debug("getMaxFromFileDyn")
    try:
        f = open(file)
    except IOError:
        logging.error("Cannot open file %s" % file)
        return None

    phys=[]
    cns=[]
    for line in f:
        if line.startswith("#"):
                continue
        # modif pour prise en compte de param phys en reel
        m = re.search("(?P<phys>\d*?[\.]\d*)\s*(?P<cn>\d+)", line)
        phy, cn = m.groups()
        phys.append(phy)
    cns.append(cn)
    physA=np.array([phys]).astype(float)
    maxphy=(physA.max().astype(float))
    return maxphy


 # quelques definitions de type de donnees pour utiliser au niveau de argparse.add_argument()
def directory(string):
    try:
        if os.path.isdir(string):
            return string
    except:
        pass
        msg = "%s directory does not exist or is not executable" % string
        raise argparse.ArgumentTypeError(msg)


def area_desc(string):
    # string est de la forme -t_srs proj4_syntax -te xmin,ymin,xmax,ymax -ts xsize,ysize
    # on l'analyse a l'aide de argparse
    p = argparse.ArgumentParser("-areadef")
    group = p.add_argument_group()
    group.add_argument('-t_srs', metavar="proj4_syntax", nargs='+', required=True)
    group.add_argument('-te', metavar=("xmin","ymin", "xmax", "ymax"), nargs=4, type=float, required=True)
    group.add_argument('-ts', metavar=("xsize","ysize"), nargs=2, type=int, required=True)
    subargs = p.parse_args(string.split())
    return {'type': 'fromcmd', 't_srs': subargs.t_srs, 'te': subargs.te, 'ts': subargs.ts}


def area(string):
    # argparse ne gere pas les groupes d'options mutuellement exclusifs
    if "@" in string:
        # def dans fichier de config
        return area_id_in_file(string)
    else:
        # def en ligne de cmd
        return area_desc(string)


# convertit un dictionnaire de paires clefs/valeurs decrivant une projection en chaine au format proj4
def dict2Proj4(dict):
    res = ""
    for k in dict.keys():
        res = "%s +%s=%s" % (res, k, dict[k])
    return res

# fonction intermediaire
def splitPath(nomVarFull):
    vartmp=nomVarFull.split('/')
    # si le chemin absolu [ /gpe1/gpe2/maVar ]  commence par /, suppression du 1er element, vide
    if vartmp[0] == '' :
        varsplit = vartmp[1:len(vartmp)]
    # si chemin absolu sous la forme gpe1/gpe2/maVar ou en reltif maVar
    else:
        varsplit = vartmp
    return varsplit


# extrait de Netcdf2pgm.java de LP && Notes..
#// -------------------------------------------------------------------------    //
#/** Calculer la correction de Li pour avoir une reflectance comme si le soleil
#*     etait au zenith. La correction est comprise entre 1 et 24. 24 pour le soleil
#*     rasant (90 deg).
#*     @param : AngleSolaire : donnees en reel en degre.
#*     @return le coefficient de correction a apporter.
#-------------------------------------------------------------------------    */
def CalculCorrectionLi(AngleSolaire):
    # AngleSolaire est une matrice
    zen = AngleSolaire.astype('float') * np.pi / 180. #// Conversion en radian.
    cos1 = np.cos(zen)

    correction_li = np.multiply(cos1,24.35 / (2. * cos1 + np.sqrt(498.5225 * cos1 * cos1 + 1.)))
        # On pourrait penser que la correction a appliquer sur l'energie solaire retournee
        # pourrait etre de 1 / cos(zen). En fait, c'est moins que ca. a cause de
        # la refraction. La correction maxi a la limite du jour et de la nuit est de 24.3.

    correction_li = np.where(correction_li < 0,0.,correction_li)

    return correction_li.astype('float')

def main():

    # parseur d'arguments
    parser = argparse.ArgumentParser(description='Reads a hdf5 file (from VIIRS instruments on SUOMI-NPP), reprojects some data and creates images as output (with tif extension)')

    # d'abord on cherche les options concernant le log qu'on initialise dans la foulee
    log_level_choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument('-loglevel', metavar='loglevel', default="INFO", choices=[item.upper() for item in log_level_choices]+[item.lower() for item in log_level_choices], help='the number of processors to use while reprojecting the data - see pyresample.kd_tree.resample_nearest for more information')
    parser.add_argument('-logfmt', metavar='logfmt', default="normal", choices=["archipel", "normal"], help='the number of processors to use while reprojecting the data - see pyresample.kd_tree.resample_nearest for more information')

    args = parser.parse_known_args()[0]

    lognumlevel = getattr(logging, args.loglevel.upper())
    try:
        logformat = args.logfmt
    except AttributeError:
        pass

    datfmt = '%d/%m/%Y %H:%M:%S'

    if logformat and logformat=="archipel":
        logfmt = '%(asctime)s:%(levelname)-7s\:(message)s'
    else:
        logfmt = '%(asctime)s\t%(levelname)-7s\t%(message)s'

    logging.basicConfig(level=lognumlevel, format=logfmt, datefmt=datfmt)

    logging.debug("Starting")
    # arguments attendus

    parser.add_argument('filePath', metavar='filePath', type=argparse.FileType('rb'), help='the path of the NetCDF file')
    parser.add_argument('imgPath', metavar='imgPath', type=directory, help='the path of the directory in which the output images will be written')

    parser.add_argument('-vrb', metavar='lstDs', help='Fichier listant les ds du fichier h5', required=True)

    parser.add_argument('-fmt', metavar='format', help='the format of the images to write out')

    parser.add_argument('-nbprocs', metavar='nbprocs', type=int, default=1, help='the number of processors to use while reprojecting the data - see pyresample.kd_tree.resample_nearest for more information')
    parser.add_argument('-fillvalue', metavar='fill_value', type=int, default=None, help='the value used to fill the areas where there is no data once the reprojection has been done - see pyresample.kd_tree.resample_nearest for more information')
    parser.add_argument('-radius', metavar='radius-of-influence', type=int, default=None, help='the radius around each grid pixel in meters to search for neighbours in the swath - see pyresample.kd_tree.resample_nearest for more information')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-nodyn', action='store_true', help='do not apply any dynamics to the data read in the NetCDF file')
    group.add_argument('-dyn', help='dynamics to apply to each variable to be reprojected in the NetCDF file, give "None" to apply no dynamics to a variable, number of arguments must be equal to the number of arguments associated with the -vrb option')
    parser.add_argument('-areadef', type=area, help='dynamics', required=False)
    parser.add_argument('-thetamax',metavar='thetamax', type=int, default=90, help='Angle maxi pour lequel la correction du visible est ecretee. Par defaut = 90 deg.')

    gp = parser.add_mutually_exclusive_group()
    gp.add_argument('-nocolorTable', action='store_true', help='do not apply any color table to the data read in the NetCDF file')
    gp.add_argument('-colorTable', metavar='colorTable',help='fichiers de la colortable [  syntaxe des lignes(nb=256) : "R,V,B,255" ]')

    # analyse de la ligne de commande
    logging.debug("Rappel de la ligne de commande")
    for item in sys.argv:
        logging.debug(item)
    logging.debug("Fin de rappel de la ligne de commande")

    args = parser.parse_args()
    logging.debug("Analyzed args: %s" % args)


    # recup de quelques elements de la ligne de commande utiles par la suite
    imgPath = args.imgPath
    filePath = args.filePath.name
    proj4_args = None
    area_extent = None
    xsize = None
    ysize = None
    areaFile = None
    areaId = None
    nbprocs = None
    fillvalue = None
    radiusofinfluence = None
    format = None
    global dynamicFiles
    dynamicFiles = None
    colorTableFiles = None
    lstDs=None

    try:
        proj4_args = args.areadef['t_srs']
    except (AttributeError, KeyError):
        pass
    try:
        area_extent = args.areadef['te']
    except (AttributeError, KeyError):
        pass
    try:
        (xsize, ysize) = args.areadef['ts']
    except (AttributeError, KeyError):
        pass
    try:
        areaFile = args.areadef['def_file']
    except (AttributeError, KeyError):
        pass
    try:
        areaId = args.areadef['area_id']
    except (AttributeError, KeyError):
        pass

    try:
        nbprocs = args.nbprocs
    except AttributeError:
        pass
    try:
        fillvalue = args.fillvalue
    except AttributeError:
        pass
    try:
        radiusofinfluence = args.radius
    except AttributeError:
        pass
    try:
        format = args.fmt
    except AttributeError:
        pass
    try:
        dynamicFiles = args.dyn.split(",")
    except AttributeError:
        pass
    try:
        thetamax = args.thetamax
    except AttributeError:
        pass
    try:
        colorTableFiles = args.colorTable.split(",")
    except AttributeError:
        pass

    noDyn=args.nodyn
    nocolorTable=args.nocolorTable
    lstDs=args.vrb

    # on initialise les variables (toutes) suivant le type de fichier
    # fichier en resol. I
    variablesName = list()
    try:
        ficListDs = open(lstDs,'r')
    except IOError:
          logging.error("Cannot open file %s" % lstDs)
          return None

    i,j=0,0
    h5fic=h5py.File(filePath,'r')

    for line in ficListDs.readlines():
        strVar="%s"% line.replace('\n','')
        if (strVar.lower().find('longitude') != -1 ):
            try:
                lons=h5fic[strVar][:,:]
                logging.info("Initialisation des longitudes avec le DS : %s" % strVar)
            except:
                logging.error("Erreur sur le ds %s [ longitude ] ABS de la source >>  verifier le fichier de config" % strVar)
        else:
            if (strVar.lower().find('latitude') != -1) :
                try:
                    lats=h5fic[strVar][:,:]
                    logging.info("Initialisation des latitudes avec le DS : %s" % strVar)
                except:
                    logging.error("Erreur sur le ds %s [ latitude ] ABS de la source : verifier le fichier de config" % strVar)
            else:
                try:
                    variablesName.append(strVar)
                    logging.info("Initialisation du DS : %s" % strVar)
                except:
                    logging.error("Erreur sur le ds %s ABS de la source : verifier le fichier de config" % strVar)
    ficListDs.close()
    logging.info("fermeture du fichier %s" % lstDs)

    logging.debug("log level : %s" % lognumlevel)
    logging.debug("img path : %s" % imgPath)
    logging.debug("file path : %s" % filePath)
    logging.debug("proj4 args : %s" % proj4_args)
    logging.debug("extent : %s" % area_extent)
    logging.debug("size : %sx%s" % (xsize, ysize))
    logging.debug("area file : %s" % areaFile)
    logging.debug("area id : %s" % areaId)
    logging.debug("nb procs : %s" % nbprocs)
    logging.debug("fill value : %s" % fillvalue)
    logging.debug("radius of influence : %s" % radiusofinfluence)
    logging.debug("dynamic files : %s" % dynamicFiles)
    logging.debug("thetamax : %s" % thetamax)
    logging.debug("colorTable Files : %s" % colorTableFiles)
    logging.info("Opening file")
    data = {}
    for i in range(len(variablesName)):
        var = variablesName[i]
        f = None
        # TODO ajouter un test de verif de presence de la variable
        data[var]=h5fic[var]
        # TODO ajouter une verif de presence des attributs suivants
        try:
            scalefactor=data[var].attrs['scale_factor']
        except :
            scalefactor=1.
            logging.warning ("Attribut scale_factor ABS du Dataset %s" % var)
        logging.info ("scale factor : %s" % scalefactor)
        try:
            offset=data[var].attrs['add_offset']
            logging.info ("offset : %s" % offset)
        except :
            offset=0.
            logging.warning("Attribut offset ABS du dataset %s" % var)

        logging.info ("offset  : %s" % offset)

        try:
            missingVal=data[var].attrs['_FillValue']
            logging.info ("missingVal : %s" % missingVal)
        except:
            missingVal=65527.
            logging.warning ("Attribut _FillValue ABS du dataset : %s" % var)

        data[var]=data[var][:,:]*scalefactor + offset

    # calcul du ndsi
    var = 'ndsi'
    scalefactorI1=h5fic['/All_Data/VIIRS-I1-SDR_All/ReflectanceFactors'][0]
    offsetI1=h5fic['/All_Data/VIIRS-I1-SDR_All/ReflectanceFactors'][1]
    data_temp_I1=data['/All_Data/VIIRS-I1-SDR_All/Reflectance'][:,:]*scalefactorI1 + offsetI1

    scalefactorI3=h5fic['/All_Data/VIIRS-I3-SDR_All/ReflectanceFactors'][0]
    offsetI3=h5fic['/All_Data/VIIRS-I3-SDR_All/ReflectanceFactors'][1]
    data_temp_I3=data['/All_Data/VIIRS-I3-SDR_All/Reflectance'][:,:]*scalefactorI3 + offsetI3

    ndviNum = (data_temp_I1[:,:]-data_temp_I3[:,:])
    ndviDen = (data_temp_I1[:,:]+data_temp_I3[:,:])
    temp = -0.01+1.45*ndviNum / ndviDen
    temp[temp<0]=0.
    temp[temp>1]=1.
    data[var]=temp
    # nettoyage
    data.pop('/All_Data/VIIRS-I1-SDR_All/Reflectance')
    data.pop('/All_Data/VIIRS-I3-SDR_All/Reflectance')

    # application de la dynamique maintenant car apres reprojection pose probleme : maskedarray
    i=0
    if noDyn or dynamicFiles and dynamicFiles[i]=="nodyn":
        pass    # pas de dynamique a appliquer
    else:
        if dynamicFiles and dynamicFiles[i]: # tentative de recup de dynamique a partir du fichier passe, si echec on genere a partir des donnees
            logging.info("Creating dynamics from file " + dynamicFiles[i])
            try:
                f=getDynamicFunctionFromFile(dynamicFiles[i])
                max=getMaxFromFileDyn(dynamicFiles[i])
                min=getMinFromFileDyn(dynamicFiles[i])
            except:
                logging.warning("-> fails")
                pass
    if not f:
        logging.info("Creating dynamics from data")
        min = data[var][data[var]>=0].min()
        max = data[var][data[var]<fillvalue].max()
        f = getDynamicFunctionFromDataNew(min,max)

    logging.info ("Info sur les donnees BRUT du fichier en entree :")
    #print data[var][0:data[var].shape[0]:data[var].shape[0]/10,0:data[var].shape[1]:data[var].shape[1]/10]

    logging.info ("MIN : %s / MAX : %s" % (data[var].min(),data[var].max()))

    # modif : filtrage des donnees abs -> revu
    selected = (data_temp_I3 < (float(missingVal)*float(scalefactorI3) + offsetI3))
    logging.info ("Pourcentage de pts corrects [ selected ] :  %s" % (100 * float(np.sum(selected==True)) / float(selected.shape[0]*selected.shape[1])))
    logging.info ("shape selected : %s %s " % selected.shape)

    data_OK = np.ma.masked_array(data[var],~selected)

    # passage en CN
    data_tmp=(f(data_OK[:,:]))
    data_tmp=data_tmp.astype('int')

    # on conserve pour l instant
    logging.info("initialisation de la grille des CNs valides")
    data_tmp_ok = np.ma.masked_array(data_OK,~selected)
    #ajout de la valeur "fillvalue" comme remplissage
    data_tmp_ok = np.ma.filled(data_tmp,fillvalue)
    logging.info ("ajout de la valeur fillvalue comme remplissage"+str(fillvalue))

    data[var]=data_tmp_ok

    # Fin d'initialisation du tableau DATA [VAR]
    logging.info("Infos sur les CN calcules : MIN : %s / MAX : %s " % (data[var].min(),data[var].max()))


    # etape de reprojection
    from pyresample import geometry

    lonsModif = np.ma.masked_array(lons,~selected)
    logging.info ("Infos sur  les longitudes filtrees (longitudes corrigees en enlevant les missingvalues ) : Min ( lonsModif ) : %s  / Max(lonsModif) : %s " % (lonsModif.min(),lonsModif.max()))

    latsModif = np.ma.masked_array(lats,~selected)
    logging.info ("Infos sur  les lattitudes filtrees (lattitudes corrigees en enlevant les missingvalues ) : Min ( latsModif ) : %s  / Max(latsModif) : %s " % (latsModif.min(),latsModif.max()))

    swath_def = geometry.SwathDefinition(lons=lonsModif, lats=latsModif)
    logging.info ("initialisation du swath_def")

    # target
    from pyresample import utils
    area_def = None

    if args.areadef['type']=="fromfile":
        logging.info("Reading %s target area definition from file %s" % (areaId, areaFile))
        try:
            area_def = utils.load_area(areaFile, areaId)
        except:
            logging.warning("-> fails")
            pass

    elif args.areadef['type']=="fromcmd":
        logging.info("Reading target area definition from command-line parameters")
        area_id = 'areaid'
        area_name = 'areaname'
        proj_id = 'projid'
        from pyresample import _spatial_mp
        proj_dict = utils._get_proj4_args(proj4_args)
        proj = _spatial_mp.Proj(**proj_dict)
        if proj.is_latlong():    # contourne le fait que pyproj travaille en radians avec les projections latlong : on souhaite passer des degres au niveau de la definition de la zone cible
            area_extent = np.radians(area_extent)
            area_def = utils.get_area_def(area_id, area_name, proj_id, proj4_args, xsize, ysize, area_extent)
            logging.debug("%s" % area_def)

    # recup des xsize, ysize, xmin, ymin, xmax, ymax de l'objet AreaDefinition pour creation du GT avec gdal-python lors de la creation du GTiff
        xsize = area_def.x_size
        ysize = area_def.y_size

        proj_dict = area_def.proj_dict
        proj = pyresample._spatial_mp.Proj(**proj_dict)
        if proj.is_latlong():    # contourne le fait que pyproj travaille en radians avec les projections latlong : on a besoin des degres pour la defition du geotransform dans gdal-python
            (xmin, ymin, xmax, ymax) = np.degrees(area_def.area_extent)
        else:
            (xmin, ymin, xmax, ymax) = area_def.area_extent


    # reproj
    from pyresample import kd_tree

    keys = (key for key, dat in data.iteritems())
    dats = (dat for key, dat in data.iteritems())
    d = np.dstack(dats)

    logging.info("Reprojection")

    result = kd_tree.resample_nearest(swath_def, d, area_def, radius_of_influence=radiusofinfluence, fill_value=fillvalue, nprocs=nbprocs)


    nbPoints = xsize * ysize

    nbptnontouche = (result[:,:, 0] == fillvalue).sum()

    nbpttouche = nbPoints - nbptnontouche
    ratioOccupation = float(nbpttouche) / float(nbPoints)

    logging.info(str(nbpttouche) + ' touched points in the domain or ' + str((int)(ratioOccupation * 100.)) + ' %')

    # plot
    from pyresample import plot
    from osgeo import gdal
    from osgeo import osr


    if not format:
        format = "GTiff"

    driver = gdal.GetDriverByName(format)

    gt = [xmin, (xmax-xmin)/xsize, 0, ymax, 0, -(ymax-ymin)/ysize]
    srs = osr.SpatialReference()
    srs.ImportFromProj4(dict2Proj4(area_def.proj_dict))
    logging.debug("wkt_srs : %s" % dict2Proj4(area_def.proj_dict))
    wkt_srs = srs.ExportToWkt()

    exts = { "gtiff" : ".tif",
        "png" : ".png",
        "jpeg" : ".jpg",
        "jpeg2000" : ".jpg",
        "gif" : ".gif",
        "pnm" : ".pnm"
        }

    i=0
    for i, k in enumerate(keys):
        try:
            logging.info("Saving result " + k)
            # a partir de GDAL-python
            tab = result[:,:,i].astype(np.dtype(np.ubyte)) # par defaut, resample retourne un tableau avec meme type que donnees initiales

            try:
                name = filename[i]
            except NameError:
                name = k

            #nameModif="%s.tif" % name.split('/')[2].replace('-SDR_All','')
            #nameModif="%s.tif" % name.split("/")[1]
            nameModif="ndsi.tif"
            dst_filename = imgPath + "/" + nameModif
            dst_ds = driver.Create( dst_filename, tab.shape[1], tab.shape[0], 1, gdal.GDT_Byte )
            dst_ds.SetGeoTransform(gt)
            dst_ds.SetProjection(wkt_srs)
            dst_ds.GetRasterBand(1).WriteArray(tab)

            # "vidage" des donnees vers le fichier produit
            dst_ds = None
            logging.info("Saved : " + dst_filename)


            # TODO : a mettre en place si utilisation par WMS
            ## ajout de la palette de couleur
            #if colorTableFiles :
                ##if colorTableFiles[i]:
                    ##print "Ajout de la palette de couleur passee en argument : %s" % colorTableFiles[i]
                    ##colorTableFile = colorTableFiles[i]
                ##else:
                #try:
                    #colorTableFile=colorTableFiles[0]
                #except:
                    #print "color Table File ABS"
                #print "Ajout de la palette de couleur par defaut : %s" %  colorTableFiles[0]


                #ct=gdal.ColorTable()
                #lst=[]
                #lines=[]
                #R=[]
                #V=[]
                #B=[]
                #ficCt = open(colorTableFile,'r')
                ## TODO ajouter un test sur la syntaxe du fichier

                #for line in ficCt.readlines():
                    #lines.append(line)
                #ficCt.close()

                #for line in lines:
                    #m = re.search("(?P<a>\d+)\,(?P<b>\d+)\,(?P<c>\d+)\,(?P<d>\d+)", line)
                    #R.append(m.groups()[0])
                    #V.append(m.groups()[1])
                    #B.append(m.groups()[2])

                #i=0
                #while (i<len(lines)):
                    #lst.append((int(R[i]),int(V[i]),int(B[i]),255))
                    #i=i+1

                #i=0
                #while i<256:
                    #ct.SetColorEntry(i,lst[i])
                    #i=i+1

                #dst_ds.GetRasterBand(1).SetColorInterpretation(2)
                #dst_ds.GetRasterBand(1).SetColorTable(ct)
                #print "--FIN Ajout de la palette de couleur"
            #if ( ( not colorTableFiles) or nocolorTable ):

                #print "pas de palette de couleurs a ajouter : IMAGE en comptes de gris"


            #dst_ds.GetRasterBand(1).WriteArray(tab)

            # "vidage" des donnees vers le fichier produit
            #dst_ds = None
        except:
            pass

    logging.info("The end !")

if __name__ == '__main__':
    start=time.time()
    main()
    end=time.time()
    logging.info("Tps de traitement  %f" % (end - start))
