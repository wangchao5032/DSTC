import random, math
from SpatialRegionTools import inregionS, inregionT

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat

def downsampling(trj_data, dropping_rate):
    noisetrip = []
    noisetrip.append(trj_data[0])
    for i in range(1, len(trj_data)-1):
        if random.random() > dropping_rate:
            noisetrip.append(trj_data[i])
    noisetrip.append(trj_data[len(trj_data)-1])
    return noisetrip

def distort(region, trip, distorting_rate, radius):
    noisetrip = []
    if region.needTime:
        for (i, [lon, lat, time]) in enumerate(trip):
            if random.random() <= distorting_rate:
                distort_dist = random.randint(0, radius)
                x, y = lonlat2meters(lon, lat)
                xnoise, ynoise = random.uniform(0,2) - 1, random.uniform(0,2) - 1
                normz = math.hypot(xnoise, ynoise)
                xnoise, ynoise = xnoise * distort_dist/normz, ynoise * distort_dist/normz
                nlon, nlat = meters2lonlat(x + xnoise, y + ynoise)
                if not (inregionT(region, nlon, nlat, time)):
                    nlon = lon
                    nlat = lat
                noisetrip.append([nlon, nlat, time])
            else:
                noisetrip.append(trip[i])
    else:
        for (i, [lon, lat]) in enumerate(trip):
            if random.random() <= distorting_rate:
                distort_dist = random.randint(0, radius)
                x, y = lonlat2meters(lon, lat)
                xnoise, ynoise = random.uniform(0,2) - 1, random.uniform(0,2) - 1
                normz = math.hypot(xnoise, ynoise)
                xnoise, ynoise = xnoise * distort_dist/normz, ynoise * distort_dist/normz
                nlon, nlat = meters2lonlat(x + xnoise, y + ynoise)
                if not (inregionS(region, nlon, nlat)):
                    nlon = lon
                    nlat = lat
                noisetrip.append([nlon, nlat])
            else:
                noisetrip.append(trip[i])
    return noisetrip

def distort_time(region, trip, distorting_time):
    noisetrip = []
    seconds = random.randint(-distorting_time, distorting_time)

    for (lon, lat, time) in trip:        
        offsettime = time + seconds
        if offsettime >= 86400:
            offsettime = offsettime - 86400
        if offsettime < 0:
            offsettime = offsettime + 86400
        if not inregionT(region, lon, lat, time):
            return trip
        noisetrip.append([lon, lat, offsettime])
    return noisetrip

def downsamplingDistort(trj_data, region):
    noisetrips = []
    dropping_rates = [0, 0.2, 0.4, 0.6]
    distorting_rates = [0, 0.2, 0.4, 0.6]
    distort_radius = 30.0
    distorting_time = 900
    for dropping_rate in dropping_rates:
        noisetrip1 = downsampling(trj_data, dropping_rate)
        if not (region.min_length <= len(noisetrip1) <= region.max_length):
            noisetrip1 = trj_data
        for distorting_rate in distorting_rates:
            noisetrip2 = distort(region, noisetrip1, distorting_rate, distort_radius)
            if region.needTime:
                noisetrip3 = distort_time(region, noisetrip2, distorting_time)
                noisetrips.append(noisetrip3)
            else:
                noisetrips.append(noisetrip2)
    return noisetrips
