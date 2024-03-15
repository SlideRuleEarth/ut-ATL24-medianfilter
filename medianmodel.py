def rolling_median_bathy_classification(point_cloud, 
                         sea_surface_indices=None,
                         window_sizes=[51, 30, 7],
                         kdiff=0.6, kstd=1.2,
                         high_low_buffer=4,
                         min_photons=14,
                         segment_length=0.001, # 0.001
                         compress_heights=None,
                         compress_lats=None):
    
    sea_surface_label = 41
    bathymetry_label = 40
    
    indices = np.arange(0, point_cloud['h_ph'].to_numpy().shape[0], 1, dtype=int)
    
    sea_surf = np.ones(point_cloud['h_ph'].to_numpy().size, dtype=bool)

    median_sea_surf = np.nanmedian(point_cloud.z_ph[sea_surface_indices])

    unique_bathy_filterlow = np.argwhere(point_cloud.z_ph > (median_sea_surf - 1.5)).flatten()
    
    mask_sea_surf = np.ones(point_cloud['h_ph'].to_numpy().size, dtype=bool)
    mask_sea_surf[sea_surface_indices] = False

    heights = point_cloud['h_ph'].to_numpy()[mask_sea_surf]
    lons = point_cloud['lon_ph'].to_numpy()[mask_sea_surf]
    lats = point_cloud['lat_ph'].to_numpy()[mask_sea_surf]
    times = point_cloud['delta_time'].to_numpy()[mask_sea_surf]

    if compress_heights is not None:
        heights = heights * compress_heights
    
    if compress_lats is not None:
        lats = lats * compress_lats

    h, lons, lats, ph_times, ind_keep = rolling_median_buffer(heights=heights, lons=lons,
                                                              lats=lats, times=times,
                                                              window_size=window_sizes[0],
                                                              high_low_buffer=high_low_buffer,
                                                              indices=indices[mask_sea_surf])

    high_ph, high_lons, high_lats, high_times, std_ind_keep = rolling_median_std(heights=h, lons=lons, lats=lats, times=ph_times,
                                                                                 keep_index=ind_keep, window_size=window_sizes[1], kdiff=kdiff, kstd=kstd)
    
    try:

        ## Rough Select Bathymetry
        rg_h_lons, rg_h_lats, rg_h_heights, rg_h_times, rg_keep_index = real_group_eliminate(lons=high_lons, lats=high_lats,
                                                                ph_h=high_ph, times=high_times, keep_index=std_ind_keep, segment_length=segment_length,
                                                                min_photons=min_photons)

        ## Average Smoothing Window
        _, _, _, _, keep_indices = rolling_average_smooth(heights=rg_h_heights,
                                                          lons=rg_h_lons,
                                                          lats=rg_h_lats,
                                                          times=rg_h_times,
                                                          keep_index=rg_keep_index,
                                                          window_size=window_sizes[2])

        classifications = np.zeros((point_cloud['h_ph'].to_numpy().shape))
        classifications[:] = 0
        
        classifications[np.asarray(keep_indices)] = bathymetry_label  # sea floor
        classifications[unique_bathy_filterlow] = 0
        classifications[sea_surface_indices] = sea_surface_label  # sea surface
        
        results = {'classification': classifications}
        
        return results
    
    except Exception as rolling_med_model_error:

        if 'cannot unpack non-iterable NoneType object' in str(rolling_med_model_error):
            print('Median Model: Failed to find bathymetry photons.')
            # print(str(traceback.format_exc()))

        # else:
        #     print(str(traceback.format_exc()))

        classifications = np.empty((point_cloud['h_ph'].to_numpy().shape))
        classifications[:] = 0
        classifications[sea_surface_indices] = sea_surface_label

        return {'classification': classifications}
            

def rolling_median_buffer(heights=None, lons=None, lats=None, times=None,
                          window_size=None, high_low_buffer=None, indices=None):

    """
        Calculates the rolling median of the heights 1D array within a defined window size.

        Based on defined high/low buffer, photons outside the median buffere range are removed.
    """

    # Adding [:,None] adds a new empty dimension to a numpy array for indexing arrays of different dims
    window_inds = np.arange(window_size) + np.arange(len(heights) - window_size + 1)[:,None]
    
    window_median = np.median(heights[window_inds], axis=1)

    high = np.unique(window_inds[heights[window_inds] > (window_median[:,None] + high_low_buffer)])
    low = np.unique(window_inds[heights[window_inds] < (window_median[:,None] - high_low_buffer)])

    ind_remove = np.unique(np.concatenate((high, low), axis=None))
    keep = np.unique(window_inds.ravel())
    ind_keep = np.delete(keep, ind_remove)

    rolling_median_heights = np.delete(heights, ind_remove)
    rolling_median_lons = np.delete(lons, ind_remove)
    rolling_median_lats = np.delete(lats, ind_remove)
    rolling_median_times = np.delete(times, ind_remove)
    indices_to_keep = np.delete(indices, ind_remove)

    return rolling_median_heights, rolling_median_lons, rolling_median_lats, rolling_median_times, indices_to_keep


def rolling_median_std(heights=None, lons=None, lats=None, times=None,
                       keep_index=None, window_size=None,
                       kdiff=None, kstd=None):

    """
        Filters elevations based on rolling median and standard deviation criteria.

        Filters elevation photons based on their deviation from the rolling median and
        the rolling standard deviation within a specified window.

    """

    # Adding [:,None] adds a new empty dimension to a numpy array for indexing arrays of different dims
    window_inds = np.arange(window_size) + np.arange(len(heights) - window_size + 1)[:,None]
    window_median = np.median(heights[window_inds], axis=1)
    kdiff_keep_inds = np.unique(window_inds[np.abs(window_median[:,None] - heights[window_inds]) < kdiff])
    window_std = np.std(heights[window_inds], axis=1, ddof=1)
    kstd_keep_inds = window_inds[(window_std < kstd)]

    comb_std_diff_keep = np.intersect1d(kdiff_keep_inds, kstd_keep_inds)

    rolling_median_heights = heights[comb_std_diff_keep]
    rolling_median_lons = lons[comb_std_diff_keep]
    rolling_median_lats = lats[comb_std_diff_keep]
    rolling_median_times = times[comb_std_diff_keep]
    
    keep_index = keep_index[comb_std_diff_keep]

    return rolling_median_heights, rolling_median_lons, rolling_median_lats, rolling_median_times, keep_index


def consecutive(data, stepsize=1):

    """
        Splits a 1D array into subarrays containing
        consecutive elements based on a specified step size.
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def real_group_eliminate(lons=None, lats=None, ph_h=None,
                         times=None, keep_index=None,
                         segment_length=None, min_photons=None):
    '''
        Group the photon heights into segments of latitude distance
        in degrees and return groups with at least min_photons. For
        groups with less than the min_photons, the heights are not
        assumed to be real.

        For the sorting to work correctly, the arrays must be
        ordered by ascending lattitude.

        segment_length of 0.001 ~ 100 m. (111m at the equator)
    '''

    if len(list(lats)) > 0:

        ## Order the arrays by ascending latitude
        if lats[0] > lats[-1]:

            lats = lats[::-1]
            lons = lons[::-1]
            ph_h = ph_h[::-1]
            times = times[::-1]

        min_lat_range = lats.min() // segment_length / (1/segment_length) + segment_length
        max_lat_range = lats.max() // segment_length / (1/segment_length)

        split_at = lats.searchsorted(np.arange(min_lat_range,
                                               max_lat_range,
                                               segment_length))

        more_than_min_photons = (split_at[1:]-split_at[:-1] > min_photons).nonzero()[0] + 1

        lat_groups = np.split(lats, split_at)
        lat_groups = np.asarray(lat_groups, dtype=object)

        lon_groups = np.split(lons, split_at)
        lon_groups = np.asarray(lon_groups, dtype=object)

        ph_h_groups = np.split(ph_h, split_at)
        ph_h_groups = np.asarray(ph_h_groups, dtype=object)

        ph_time_groups = np.split(times, split_at)
        ph_time_groups = np.asarray(ph_time_groups, dtype=object)
        
        keep_index_groups = np.split(keep_index, split_at)
        keep_index_groups = np.asarray(keep_index_groups, dtype=object)

        try:

            grouped_lons = [np.concatenate([lon_groups[i]][0]).ravel() for i in consecutive(more_than_min_photons, stepsize=1)]
            grouped_lats = [np.concatenate([lat_groups[i]][0]).ravel() for i in consecutive(more_than_min_photons, stepsize=1)]
            grouped_ph_h = [np.concatenate([ph_h_groups[i]][0]).ravel() for i in consecutive(more_than_min_photons, stepsize=1)]
            grouped_ph_times = [np.concatenate([ph_time_groups[i]][0]).ravel() for i in consecutive(more_than_min_photons, stepsize=1)]
            grouped_keep_index_groups = [np.concatenate([keep_index_groups[i]][0]).ravel() for i in consecutive(more_than_min_photons, stepsize=1)]

            return grouped_lons, grouped_lats, grouped_ph_h, grouped_ph_times, grouped_keep_index_groups

        except Exception as medmodel_error:

            if 'need at least one array to concatenate' in str(medmodel_error):
                print('Median Model: No bathymetry photons found in this segment')

            # else:
            #     print(str(traceback.format_exc()))

            return None
    else:
        return None


def flatten_coord_blocks(coord_block_array=None):
    return [item for block in coord_block_array for item in block]


def rolling_average_smooth(heights=None, lons=None, lats=None,
                           times=None, keep_index=None, window_size=None):
    """
        Performs a rolling average smoothing along photon elevations with window_size.

        This function calculates the rolling average of a 1D array within a specified
        window_size.
    """

    if (window_size % 2) == 0:
        raise Exception('window_size must be odd')

    # Adding [:,None] adds a new empty dimension to a numpy array for indexing arrays of different dims
    window_inds = [np.arange(window_size) + np.arange(len(heights[i]) - window_size + 1)[:,None] for i in np.arange(len(heights))]
    window_mean = [np.mean(heights[i][window_inds[i]], axis=1) for i in np.arange(len(heights))]
    window_centers = [window_inds[i][:,(window_size // 2)] for i in np.arange(len(heights))]

    center_lats = [lats[i][window_centers[i]] for i in np.arange(len(heights))]
    center_lons = [lons[i][window_centers[i]] for i in np.arange(len(heights))]
    center_times = [times[i][window_centers[i]] for i in np.arange(len(heights))]
    center_inds = [keep_index[i][window_centers[i]] for i in np.arange(len(heights))]

    return flatten_coord_blocks(center_lons), flatten_coord_blocks(center_lats), \
           flatten_coord_blocks(window_mean), flatten_coord_blocks(center_times), \
           flatten_coord_blocks(center_inds)



def time2UTC(gps_seconds_array=None):

    # Number of Leap seconds
    # See: 'https://www.ietf.org/timezones/data/leap-seconds.list'
    # 15 - 1 Jan 2009
    # 16 - 1 Jul 2012
    # 17 - 1 Jul 2015
    # 18 - 1 Jan 2017
    leap_seconds = 18

    gps_start = datetime.datetime(year=1980, month=1, day=6)
    time_ph = [datetime.timedelta(seconds=time) for time in gps_seconds_array]
    # last_photon = datetime.timedelta(seconds=gps_seconds[-1])
    error = datetime.timedelta(seconds=leap_seconds)
    ph_time_utc = [(gps_start + time - error) for time in time_ph]

    return ph_time_utc


















##################################
##################################
##################################
##################################
##################################

##################################











def binned_processing(pointcloud=None, window_size=None, sea_surface_label=None):
    """
        Process a given profile to produce a model.

        Args:
            profile: The profile object to be processed.

        Returns:
            Model: A Model object containing the processed data.
    """
    step_along_track = 1

    range_z = (-100, 100)
    res_z = 0.5
    res_along_track = 100

    z_min = range_z[0]
    z_max = range_z[1]
    z_bin_count = np.int64(np.ceil((z_max - z_min) / res_z))
    bin_edges_z = np.linspace(z_min, z_max, num=z_bin_count+1)

    data = copy.deepcopy(pointcloud)

    # along track bin sizing
    #   get the max index of the dataframe
    #   create bin group ids (at_idx_grp) based on pho_count spacing
    at_max_idx = data.x_ph.max()
    at_min_idx = data.x_ph.min()
    at_idx_grp = np.arange(at_min_idx, at_max_idx + res_along_track, res_along_track)
    
    # sort the data by distnace along track, reset the index
    # add 'at_grp' column for the bin group id
    data.sort_values(by='x_ph', inplace=True)
    #data.reset_index(inplace=True)
    data['idx'] = data.index
    data['at_grp'] = 0

    # for each bin group, assign an interger id for index values between each of the 
    #   group bin values. is pho_count = 20 then at_idx_grp = [0,20,49,60...]
    #   - data indicies between 0-20: at_grp = 1
    #   - data indicies between 20-40: at_grp = 2...

    pd.options.mode.chained_assignment = None

    for i, grp in enumerate(at_idx_grp):
        if grp < at_idx_grp.max():
            data['at_grp'][data['x_ph'].between(at_idx_grp[i], at_idx_grp[i+1])] = (at_idx_grp[i] - at_idx_grp.min()) / res_along_track

    pd.options.mode.chained_assignment = 'warn'
    
    # add group bin columns to the profile, photon group bin index
    data['pho_grp_idx'] = data['at_grp']
    
    # calculating the range so the histogram output maintains exact along track values
    at_min = data.x_ph.min()
    xh = (data.x_ph.values)
    bin_edges_at_min = data.groupby('at_grp').x_ph.min().values
    bin_edges_at_max = data.groupby('at_grp').x_ph.max().values

    bin_edges_at = np.concatenate([np.array([data.x_ph.min()]), bin_edges_at_max])

    # array to store actual interpolated model and fitted model
    hist_modeled = np.nan * np.zeros((bin_edges_at.shape[0] -1, z_bin_count))
    
    start_step = (window_size) / 2
    end_step = len(bin_edges_at)

    # -1 to start index at 0 instead of 1. For correct indexing when writing to hist_modeled array.
    win_centers = np.arange(np.int64(start_step), np.int64(
        end_step), step_along_track) -1

    window_args = create_window_processing_args(data=data, window_size=window_size, win_centers=win_centers, bin_edges_at=bin_edges_at)
    

    # print('Starting Parallel Processing')
    parallel = True
    if parallel:

        processed_data = []
        with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
            processed_data = pool.starmap(process_window,
                                          zip(repeat(window_size),
                                              window_args['window_centres'],
                                                    repeat(z_min), repeat(z_max),
                                                    repeat(z_bin_count), repeat(res_z),
                                                    window_args['window_profiles'],
                                                    repeat(bin_edges_z), window_args['at_begs'],
                                                    window_args['at_ends']))
    
        replace_indices = np.hstack([elem[0] for elem in processed_data if elem[0] is not None])
        sea_surface = np.hstack([np.full(elem[0].shape[0], elem[1]) for elem in processed_data if elem[0] is not None])

        # print('replace_indices: ',replace_indices)
        # print('sea_surface: ',sea_surface)

        # sea_surface_values = np.zeros(pointcloud['ph_index'].to_numpy().shape[0])
        sea_surface_classifications = np.full(pointcloud['ph_index'].to_numpy().shape[0], sea_surface_label)
        classifications = np.zeros(pointcloud['ph_index'].to_numpy().shape[0])

        np.put(classifications, replace_indices, sea_surface_classifications)

        # np.put(sea_surface_values, replace_indices, sea_surface_label)
        pointcloud['classifications'] = classifications

        # for replace_index, ssurf in zip(replace_indices, sea_surface):

        #     pointcloud.loc[pointcloud['ph_index'] == replace_index] = ssurf
            # update_DF(df=pointcloud, col_ind_val=replace_indices, update_val=sea_surface)

            # def update_DF(df=None, col_ind_val=None, update_val=None):
            #     df.loc[df['ph_index'] == col_ind_val] = update_val
            #     return df

    #     replace_with = np.vstack([np.flip(elem[1]) for elem in processed_data])

    #     np.put_along_axis(hist_modeled, replace_indices, replace_with, axis=0)\
        
    return pointcloud




def create_window_processing_args(data=None, window_size=None, win_centers=None, bin_edges_at=None):


    at_begs = []
    at_ends = []
    window_profiles = []
    window_centres = []

    xh = (data.x_ph.values)

    # if photon_bins == False: 

    for window_centre in win_centers:

        # get indices/at distance of evaluation window
        i_beg = np.int64(max((window_centre - (window_size-1) / 2), 0))
        i_end = np.int64(min((window_centre + (window_size-1) / 2), len(bin_edges_at)-2)) + 1
        # + 1 above pushes i_end to include up to the edge of the bin when used as index

        at_beg = bin_edges_at[i_beg]
        at_end = bin_edges_at[i_end]

        # could be sped up with numba if this is an issue
        # subset data using along track distance window
        i_cond = ((xh > at_beg) & (xh < at_end))

        # copy profile to avoid overwriting
        w_profile = copy.deepcopy(data)

        w_profile = w_profile.loc[i_cond, :]

        # # remove all data except for the photons in this window
        # w_profile = df_win

        at_begs.append(at_beg)
        at_ends.append(at_end)
        window_profiles.append(w_profile)
        window_centres.append(window_centre)
  
    return {'at_begs': at_begs,
            'at_ends': at_ends,
            'window_profiles': window_profiles,
            'window_centres': window_centres}

def process_window(window_size=None, window_centre=None,
                   z_min=None, z_max=None, z_bin_count=None,
                   res_z=None,
                   win_profile=None, bin_edges_z=None,
                   at_beg=None, at_end=None):

    # version of dataframe with only nominal photons
    # use this data for constructing waveforms
    df_win_nom = win_profile.loc[win_profile.quality_ph == 0]

    height_geoid = df_win_nom.z_ph.values

    # subset of histogram data in the evaluation window
    h_ = histogram1d(height_geoid,
                        range=[z_min, z_max], bins=z_bin_count)

    # smooth the histogram with 0.2 sigma gaussian kernel
    h = gaussian_filter1d(h_, 0.2/res_z)

    # identify median lat lon values of photon data in this chunk
    # x_win = df_win_nom.lon_ph.median()
    # y_win = df_win_nom.lat_ph.median()
    # at_win = df_win_nom.x_ph.median()
    # any_sat_ph = (win_profile.quality_ph > 0).any()

    photon_inds, sea_surf = find_sea_surface(hist=np.flip(h), z_inv=-np.flip(bin_edges_z), win_profile=win_profile)

    return photon_inds, sea_surf

















def find_sea_surface(hist=None, z_inv=None, win_profile=None):

    peak_info = get_peak_info(hist, z_inv)

    z_bin_size = np.unique(np.diff(z_inv))[0]
    zero_val = 1e-31
    quality_flag = False
    params_out = {}
    # ########################## ESTIMATING SURFACE PEAK PARAMETERS #############################

    # Surface return - largest peak
    peak_info.sort_values(by='prominences', inplace=True, ascending=False)


    if peak_info.empty:
        return None, None

    # if the top two peaks are within 20% of each other
    # go with the higher elev one/the one closest to 0m
    # mitigating super shallow, bright reefs where seabed is actually brighter

    if peak_info.shape[0] > 1:

        # check if second peak is greater than 20% the prominence of the primary
        two_tall_peaks = (
            (peak_info.iloc[1].prominences) / peak_info.iloc[0].prominences) > 0.2

        if two_tall_peaks:
            # use the peak on top of the other
            # anywhere truly open water will have the surface as the top peak
            pks2 = peak_info.iloc[[0, 1]]
            peak_above = pks2.z_inv.argmin()
            surf_pk = peak_info.iloc[peak_above]

        else:
            surf_pk = peak_info.iloc[0]
    else:

        # print('peak_info: ', peak_info)
        surf_pk = peak_info.iloc[0]

    # estimate noise rates above surface peak and top of surface peak

    # dont use ips, it will run to the edge of an array with a big peak
    # using std estimate of surface peak instead
    surface_peak_left_edge_i = np.int64(
        np.floor(surf_pk.i - 2 * surf_pk.sigma_est_left_i))

    # distance above surface peak to start estimating noise

    height_above_surf_noise_est = 30 # meters

    above_surface_noise_left_edge_i = surface_peak_left_edge_i - (height_above_surf_noise_est / z_bin_size)

    # # if theres not 15m above surface, use what there is above the surface
    if above_surface_noise_left_edge_i <= 0:
        above_surface_noise_left_edge_i = 0

    above_surface_idx_range = np.arange(above_surface_noise_left_edge_i, surface_peak_left_edge_i, dtype=np.int64)

    # # above surface estimation
    if surface_peak_left_edge_i <= 0:
        # no bins above the surface
        params_out['background'] = zero_val

    else:
        # median of all bins 15m above the peak
        params_out['background'] = np.median(
            hist[above_surface_idx_range]) + zero_val  # eps to avoid 0

    # use above surface data to refine the estimate of the top of the surface
    # top of the surface is the topmost surface bin with a value greater than the noise above
    
    # photons through the top of the surface
    top_to_surf_center_i = np.arange(surf_pk.i+1, dtype=np.int64)
    less_than_noise_bool = hist[top_to_surf_center_i] <= (2 * params_out['background'])
    # found just using the noise estimate x 1 leads to too wide of a window in quick testing, using x2 for now

    # get the lowest elevation bin higher than the above surface noise rate
    # set as the top of the surface

    if less_than_noise_bool.any():
        surface_peak_left_edge_i = np.where(less_than_noise_bool)[0][-1]
    
    # else defaults to 3 sigma range defined above

    # but this is a rough estimate of the surface peak assuming perfect gaussian on peak
    # surface peak is too important to leave this rough estimate
    # surf peak can have a turbid subsurface tail, too

    # elbow detection - where does the surface peak end and become subsurface noise/signal?
    # basically mean value theorem applied to the subsurface histogram slope
    # how to know when the histogram starts looking like possible turbidity?
    # we assume the surface should dissipate after say X meters depth, but will sooner in actuality
    # therefore, the actual numerical slope must cross the threshold required to dissipate after X meters at some point
    # where X is a function of surface peak width

    # what if the histogram has a rounded surface peak, or shallow bathy peaks?
    # rounded surface peak would be from pos to neg (excluded)
    # shallow bathy peaks would be the 2nd or 3rd slope threshold crossing
    # some sketches with the curve and mean slope help show the sense behind this

    # we'll use this to improve our estimate of the surface gaussian early on
    # elbow detection start
    dissipation_range = 10  # m # was originally 3m but not enough for high turbid cases
    slope_thresh = -surf_pk.heights / (dissipation_range/z_bin_size)
    diffed_subsurf = np.diff(hist[np.int64(surf_pk.i):])

    # detection of slope decreasing in severity, crossing the thresh
    sign_ = np.sign(diffed_subsurf - slope_thresh)
    # ie where sign changes (from negative to positive only)
    sign_changes_i = np.where(
        (sign_[:-1] != sign_[1:]) & (sign_[:-1] < 0))[0] + 1

    # if len(sign_changes_i) == 0:
    #     no_sign_change = True
    #     # basically aa surface at the bottom of window somehow
    #     quality_flag = -4
    #     # params_out = pd.Series(params_out, name='params', dtype=np.float64)
    #     bathy_quality_ratio = -1
    #     surface_gm = None 
    #     return params_out, quality_flag, bathy_quality_ratio, surface_gm

    # else:
    # calculate corner details
    transition_i = np.int64(surf_pk.i) + \
        sign_changes_i[0]  # full hist index

    # bottom bound of surface peak in indices
    surface_peak_right_edge_i = transition_i + 1
    # params_out['column_top'] = -z_inv[surface_peak_right_edge_i]

    # end elbow detection

    if surface_peak_right_edge_i > len(hist):
        surface_peak_right_edge_i = len(hist)


    surf_range_i = np.arange(surface_peak_left_edge_i,
                             surface_peak_right_edge_i, dtype=np.int64)
    
    clipped_prof = win_profile.loc[(win_profile['z_ph'] < -z_inv[surface_peak_left_edge_i]) & (win_profile['z_ph'] > -z_inv[surface_peak_right_edge_i])]
    z_ph = win_profile.z_ph
    z_surf = z_ph[(z_ph < -z_inv[surface_peak_left_edge_i]) & (z_ph > -z_inv[surface_peak_right_edge_i])]
    z_inv_surf = -z_surf

    return clipped_prof['ph_index'].to_numpy(), z_surf.median()
    


def get_peak_info(hist, z_inv, verbose=False):
    """
    Evaluates input histogram to find peaks and associated peak statistics.
    
    Args:
        hist (array): Histogram of photons by z_inv.
        z_inv (array): Centers of z_inv bins used to histogram photon returns.
        verbose (bool, optional): Option to print output and warnings. Defaults to False.

    Returns:
        Pandas DataFrame with the following columns:
            - i: Peak indices in the input histogram.
            - prominences: Prominence of the detected peaks.
            - left_bases, right_bases: Indices of left and right bases of the peaks.
            - left_z, right_z: z_inv values of left and right bases of the peaks.
            - heights: Heights of the peaks.
            - fwhm: Full Width at Half Maximum of the peaks.
            - left_ips_hm, right_ips_hm: Left and right intersection points at half maximum.
            - widths_full: Full width of peaks.
            - left_ips_full, right_ips_full: Left and right intersection points at full width.
            - sigma_est_i: Estimated standard deviation indices.
            - sigma_est: Estimated standard deviation in units of z_inv.
            - prom_scaling_i, mag_scaling_i: Scaling factors for prominences and magnitudes using indices.
            - prom_scaling, mag_scaling: Scaling factors for prominences and magnitudes in units of z_inv.
            - z_inv: z_inv values at the peak indices.
    """
    
    z_inv_bin_size = np.unique(np.diff(z_inv))[0]

    # padding elevation mapping for peak finding at edges
    z_inv_padded = z_inv

    # left edge
    z_inv_padded = np.insert(z_inv_padded,
                             0,
                             z_inv[0] - z_inv_bin_size)  # use zbin for actual fcn

    # right edge
    z_inv_padded = np.insert(z_inv_padded,
                             len(z_inv_padded),
                             z_inv[-1] + z_inv_bin_size)  # use zbin for actual fcn

    dist_req_between_peaks = 0.49999  # m

    if dist_req_between_peaks/z_inv_bin_size < 1:
        warn_msg = '''get_peak_info(): Vertical bin resolution is greater than the req. min. distance 
        between peak. Setting req. min. distance = z_inv_bin_size. Results may not be as expected.
        '''
        if verbose:
            warnings.warn(warn_msg)
        dist_req_between_peaks = z_inv_bin_size

    # note: scipy doesnt identify peaks at the start or end of the array
    # so zeros are inserted on either end of the histogram and output indexes adjusted after

    # distance = distance required between peaks - use approx 0.5 m, accepts floats >=1
    # prominence = required peak prominence
    pk_i, pk_dict = find_peaks(np.pad(hist, 1),
                               distance=dist_req_between_peaks/z_inv_bin_size,
                               prominence=0.01)

    # evaluating widths with find_peaks() seems to be using it as a threshold - not desired
    # width = required peak width (index) - use 1 to return all
    # rel_height = how far down the peak to measure its width
    # 0.5 is half way down, 1 is measured at the base
    # approximate stdm from the full width and half maximum
    pk_dict['fwhm'], pk_dict['width_heights_hm'], pk_dict['left_ips_hm'], pk_dict['right_ips_hm'] \
        = peak_widths(np.pad(hist, 1), pk_i, rel_height=0.4)

    # calculate widths at full prominence, more useful than estimating peak width by std
    pk_dict['widths_full'], pk_dict['width_heights_full'], pk_dict['left_ips_full'], pk_dict['right_ips_full'] \
        = peak_widths(np.pad(hist, 1), pk_i, rel_height=1)

    # organize into dataframe for easy sorting and reindex
    pk_dict['i'] = pk_i - 1
    pk_dict['heights'] = hist[pk_dict['i']]

    # draw a horizontal line at the peak height until it cross the signal again
    # min values within that window identifies the bases
    # preference for closest of repeated minimum values
    # ie. can give weird values to the left/right of the main peak, and to the right of a bathy peak
    # when theres noise in a scene with one 0 bin somewhere far
    pk_dict['left_z'] = z_inv_padded[pk_dict['left_bases']]
    pk_dict['right_z'] = z_inv_padded[pk_dict['right_bases']]
    pk_dict['left_bases'] -= 1
    pk_dict['right_bases'] -= 1

    # left/right ips = interpolated positions of left and right intersection points
    # of a horizontal line at the respective evaluation height.
    # mapped to input indices so needs adjustingpk_dict['left_ips'] -= 1
    pk_dict['right_ips_hm'] -= 1
    pk_dict['left_ips_hm'] -= 1
    pk_dict['right_ips_full'] -= 1
    pk_dict['left_ips_full'] -= 1

    # estimate gaussian STD from the widths
    # sigma estimate in terms of int indexes
    pk_dict['sigma_est_i'] = (pk_dict['fwhm'] / 2.35)
    # sigma estimate in terms of int indexes
    pk_dict['sigma_est_left_i'] = (
        2*(pk_dict['i'] - pk_dict['left_ips_hm']) / 2.35)
    # sigma estimate in terms of int indexes
    pk_dict['sigma_est_right_i'] = (
        2*(pk_dict['right_ips_hm'] - pk_dict['i']) / 2.35)

    # sigma estimate in terms of int indexes
    pk_dict['sigma_est'] = z_inv_bin_size * (pk_dict['fwhm'] / 2.35)

    # approximate gaussian scaling factor based on prominence or magnitudes
    # for gaussians range indexed
    pk_dict['prom_scaling_i'] = pk_dict['prominences'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est_i'])
    pk_dict['mag_scaling_i'] = pk_dict['heights'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est_i'])

    # for gaussians mapped to z
    pk_dict['prom_scaling'] = pk_dict['prominences'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est'])
    pk_dict['mag_scaling'] = pk_dict['heights'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est'])
    pk_dict['z_inv'] = z_inv[pk_dict['i']]

    peak_info = pd.DataFrame.from_dict(pk_dict, orient='columns')
    peak_info.sort_values(by='prominences', inplace=True, ascending=False)

    return peak_info



def first_pass_sea_surface(pointcloud=None, sea_surface_label=None):

    classified_pointcloud = binned_processing(pointcloud, window_size=3, sea_surface_label=sea_surface_label)

    return classified_pointcloud



def plot_pointcloud(classified_pointcloud=None, output_path=None):

    import matplotlib as mpl
    from matplotlib import pyplot as plt

    ylim_min = -80
    ylim_max = 20

    # xlim_min = 24.5
    # xlim_max = 25

    plt.figure(figsize=(48, 16))
    
    plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['classifications'] == 0.0],
                classified_pointcloud['z_ph'][classified_pointcloud['classifications'] == 0.0],
                'o', color='0.7', label='Noise', markersize=2, zorder=1)
    
    plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['classifications'] == 41.0],
                classified_pointcloud['z_ph'][classified_pointcloud['classifications'] == 41.0],
                'o', color='blue', label='Sea Surface', markersize=5, zorder=5)
    
    plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['classifications'] == 40.0],
                classified_pointcloud['z_ph'][classified_pointcloud['classifications'] == 40.0],
                'o', color='red', label='Bathymetry', markersize=5, zorder=5)

#         plt.scatter(point_cloud.x[point_cloud._bathy_classification_counts == 1],
#                  point_cloud.z[point_cloud._bathy_classification_counts == 1],
#                  s=1, marker='.', c=point_cloud._bathy_classification_counts[point_cloud._bathy_classification_counts == 1], cmap='cool', vmin=0, vmax=1, label='Seabed')
    # if point_cloud._z_refract is not None:
    #     if point_cloud._z_refract.any():
    #         plt.scatter(point_cloud.y[point_cloud._bathy_classification_counts > 0],
    #             point_cloud._z_refract[point_cloud._bathy_classification_counts > 0],
    #             s=36, marker='o', c=point_cloud._bathy_classification_counts[point_cloud._bathy_classification_counts > 0], cmap='Reds', vmin=0, vmax=1, label='Refraction Corrected', zorder=11)

    plt.xlabel('Latitude (degrees)', fontsize=36)
    plt.xticks(fontsize=34)
    plt.ylabel('Height (m)', fontsize=36)
    plt.yticks(fontsize=34)
    plt.ylim(ylim_min, ylim_max)
    # plt.xlim(xlim_min, xlim_max)
    plt.title('Med Filter Predictions', fontsize=40)
    # plt.title(fname + ' ' + channel)
    plt.legend(fontsize=36)
    
    plt.savefig(output_path + '_FINAL.png')
    plt.close()

    

    return


# def plot_pointcloud_QF(classified_pointcloud=None, output_path=None):

#     import matplotlib as mpl
#     from matplotlib import pyplot as plt

#     ylim_min = -80
#     ylim_max = 20

#     # xlim_min = 24.5
#     # xlim_max = 25

#     plt.figure(figsize=(48, 16))
    
#     plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['quality_ph'] == 0.0],
#                 classified_pointcloud['z_ph'][classified_pointcloud['quality_ph'] == 0.0],
#                 'o', color='green', label='quality_ph = 0 (nominal)', markersize=2, zorder=1)
    
#     plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['quality_ph'] == 1.0],
#                 classified_pointcloud['z_ph'][classified_pointcloud['quality_ph'] == 1.0],
#                 'o', color='blue', label='quality_ph = 1 (possible_afterpulse)', markersize=5, zorder=5)
    
#     plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['quality_ph'] == 2.0],
#                 classified_pointcloud['z_ph'][classified_pointcloud['quality_ph'] == 2.0],
#                 'o', color='red', label='quality_ph = 2 (possible_impulse_response)', markersize=5, zorder=5)

#     plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['quality_ph'] == 3.0],
#                 classified_pointcloud['z_ph'][classified_pointcloud['quality_ph'] == 3.0],
#                 'o', color='red', label='quality_ph = 3 (possible_tep)', markersize=5, zorder=5)

# #         plt.scatter(point_cloud.x[point_cloud._bathy_classification_counts == 1],
# #                  point_cloud.z[point_cloud._bathy_classification_counts == 1],
# #                  s=1, marker='.', c=point_cloud._bathy_classification_counts[point_cloud._bathy_classification_counts == 1], cmap='cool', vmin=0, vmax=1, label='Seabed')
#     # if point_cloud._z_refract is not None:
#     #     if point_cloud._z_refract.any():
#     #         plt.scatter(point_cloud.y[point_cloud._bathy_classification_counts > 0],
#     #             point_cloud._z_refract[point_cloud._bathy_classification_counts > 0],
#     #             s=36, marker='o', c=point_cloud._bathy_classification_counts[point_cloud._bathy_classification_counts > 0], cmap='Reds', vmin=0, vmax=1, label='Refraction Corrected', zorder=11)

#     plt.xlabel('Latitude (degrees)', fontsize=36)
#     plt.xticks(fontsize=34)
#     plt.ylabel('Height (m)', fontsize=36)
#     plt.yticks(fontsize=34)
#     plt.ylim(ylim_min, ylim_max)
#     # plt.xlim(xlim_min, xlim_max)
#     plt.title('Quality flags', fontsize=40)
#     # plt.title(fname + ' ' + channel)
#     plt.legend(fontsize=36)
    
#     plt.savefig(output_path + '_FINAL_quality_flags.png')
#     plt.close()

    

#     return


def plot_pointcloud_truth_comparison(classified_pointcloud=None, output_path=None):

    import matplotlib as mpl
    from matplotlib import pyplot as plt

    f, ax = plt.subplots(2, 1, figsize=(48,16), 
                                 sharex=True)

    ylim_min = -80
    ylim_max = 20


    # classified_pointcloud = classified_pointcloud.loc[classified_pointcloud['quality_ph'] == 0]
    
    ax[0].plot(classified_pointcloud['lat_ph'][classified_pointcloud['classifications'] == 0.0],
                classified_pointcloud['z_ph'][classified_pointcloud['classifications'] == 0.0],
                'o', color='0.7', label='Noise', markersize=2, zorder=1)
    
    ax[0].plot(classified_pointcloud['lat_ph'][classified_pointcloud['classifications'] == 41.0],
                classified_pointcloud['z_ph'][classified_pointcloud['classifications'] == 41.0],
                'o', color='blue', label='Sea Surface', markersize=5, zorder=5)
    
    ax[0].plot(classified_pointcloud['lat_ph'][classified_pointcloud['classifications'] == 40.0],
                classified_pointcloud['z_ph'][classified_pointcloud['classifications'] == 40.0],
                'o', color='red', label='Bathymetry', markersize=5, zorder=5)
    

    ax[1].plot(classified_pointcloud['lat_ph'][classified_pointcloud['manual_label'] == 0.0],
                classified_pointcloud['z_ph'][classified_pointcloud['manual_label'] == 0.0],
                'o', color='0.7', label='Noise', markersize=2, zorder=1)
    
    ax[1].plot(classified_pointcloud['lat_ph'][classified_pointcloud['manual_label'] == 41.0],
                classified_pointcloud['z_ph'][classified_pointcloud['manual_label'] == 41.0],
                'o', color='blue', label='Sea Surface', markersize=5, zorder=5)
    
    ax[1].plot(classified_pointcloud['lat_ph'][classified_pointcloud['manual_label'] == 40.0],
                classified_pointcloud['z_ph'][classified_pointcloud['manual_label'] == 40.0],
                'o', color='green', label='Bathymetry', markersize=5, zorder=5)

    ax[0].set_xlabel('Latitude (degrees)', fontsize=36)
    ax[0].set_ylabel('Height (m)', fontsize=36)
    ax[0].tick_params(axis='x', labelsize=34)
    ax[0].tick_params(axis='y', labelsize=34)
    ax[0].set_ylim(ylim_min, ylim_max)
    # ax[0].set_xlim(xlim_min, xlim_max)
    ax[0].set_title('Med Filter Predictions', fontsize=40)
    # plt.title(fname + ' ' + channel)
    ax[0].legend(fontsize=36)

    ax[1].set_xlabel('Latitude (degrees)', fontsize=36)
    ax[1].set_ylabel('Height (m)', fontsize=36)
    ax[1].tick_params(axis='x', labelsize=34)
    ax[1].tick_params(axis='y', labelsize=34)
    ax[1].set_ylim(ylim_min, ylim_max)
    # ax[1].set_xlim(xlim_min, xlim_max)
    ax[1].set_title('Truth Manual Labels', fontsize=40)
    # plt.title(fname + ' ' + channel)
    ax[1].legend(fontsize=36)
    
    f.subplots_adjust(hspace=0.4)
    f.savefig(output_path + '_FINAL_truth_comparison.png')
    plt.close(f)



    return


def main(args):

    input_fname = args.photon_data_fname
    output_label_fname = args.output_label_fname

    sea_surface_label = 41
    bathymetry_label = 40

    point_cloud = pd.read_csv(input_fname)

    point_cloud['ph_index'] = np.arange(0, point_cloud.shape[0], 1)

    
    # Used for temporary sea surface classification.

    point_cloud['dist_ph_along_total'] = point_cloud['segment_dist_x'] + point_cloud['dist_ph_along']
    point_cloud['x_ph'] = point_cloud['dist_ph_along_total'] - point_cloud['dist_ph_along_total'].min()
    
    # calculate geodetic heights
    #   ellipsoidal height (heights/h_ph) - geoid (geophys/geoid)
    point_cloud['z_ph'] = point_cloud['h_ph'] - point_cloud['geoid']
    point_cloud['delta_time'] = time2UTC(gps_seconds_array=point_cloud['gps_seconds'].to_numpy())

    classified_pointcloud = first_pass_sea_surface(pointcloud=point_cloud,
                                                sea_surface_label=sea_surface_label)
    
    class_arr = classified_pointcloud['classifications'].to_numpy()
    sea_surface_inds = np.argwhere(class_arr == sea_surface_label).flatten()



    # Start Bathymetry Classification


    plot_path = output_label_fname.replace('.csv', '.png')

    # plot_pointcloud(classified_pointcloud=classified_pointcloud, output_path=plot_path)

    rolling_median_filter_results = rolling_median_bathy_classification(point_cloud=point_cloud,
                                                                        window_sizes=[51, 30, 7],
                                                                        kdiff=0.8, kstd=1.4,
                                                                        high_low_buffer=4,
                                                                        min_photons=14,
                                                                        segment_length=0.001,
                                                                        compress_heights=None,
                                                                        compress_lats=None,
                                                                        sea_surface_indices=sea_surface_inds)

    point_cloud['classifications'] = rolling_median_filter_results['classification']

    plot_pointcloud(classified_pointcloud=point_cloud, output_path=plot_path)

    # plot_pointcloud_QF(classified_pointcloud=classified_pointcloud, output_path=plot_path)

    plot_pointcloud_truth_comparison(classified_pointcloud=point_cloud, output_path=plot_path)

    # classified_pointcloud.to_csv(output_label_fname)

    # except Exception as classify_run_error:

    #     print('classify_run_error: ', classify_run_error)
    #     print(str(traceback.format_exc()))

    #     print('Failed file: ', input_fname)

    #     failed_classification_array = np.zeros(point_cloud.shape[0])

    #     failed_classification_df = pd.DataFrame(data={'classifications': failed_classification_array})

    #     point_cloud['classifications'] = failed_classification_array

    #     plot_path = output_label_fname.replace('.csv', '_FAILED.png')

    #     print('Failed plot path: ', plot_path)


    #     plot_pointcloud_truth_comparison(classified_pointcloud=point_cloud, output_path=plot_path)

    #     print('Writing Array')

    #     # failed_classification_df.to_csv(output_label_fname)
        


    
    # label_df.to_csv(output_label_fname)







if __name__=="__main__":

    import argparse
    import numpy as np
    import datetime
    import traceback
    import pandas as pd

    import multiprocessing
    from itertools import repeat
    import copy
    import traceback

    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, peak_widths
    from fast_histogram import histogram2d, histogram1d

    parser = argparse.ArgumentParser()

    parser.add_argument("--photon-data-fname", default=True)
    parser.add_argument("--output-label-fname", default=True)

    args = parser.parse_args()

    
    main(args)



