// Helper functions for data operations and API calls

export const performDataLoad = async (operationId, options) => {
  const {
    apiClient,
    apiMethod = 'get',
    apiEndpoint,
    requestData,
    dataPath,
    onBeforeStart,
    onSuccess,
    onError,
    successMessage,
    errorMessage,
    errorContext,
    showSuccess = true,
    showError = true,
    notify,
    handleError,
    fallbackValue,
    apiCall // Support apiCall parameter for direct API function calls
  } = options;

  try {
    // Call before start callback
    if (onBeforeStart) {
      onBeforeStart();
    }

    // Make API call - support both apiCall function and traditional apiClient + apiEndpoint
    let response;
    if (apiCall && typeof apiCall === 'function') {
      // Use apiCall function if provided
      response = await apiCall();
    } else if (apiClient && apiEndpoint) {
      // Traditional apiClient + apiEndpoint approach
      if (apiMethod === 'get') {
        response = await apiClient.get(apiEndpoint);
      } else if (apiMethod === 'post') {
        response = await apiClient.post(apiEndpoint, requestData);
      } else if (apiMethod === 'put') {
        response = await apiClient.put(apiEndpoint, requestData);
      } else if (apiMethod === 'delete') {
        response = await apiClient.delete(apiEndpoint);
      } else {
        throw new Error(`Unsupported API method: ${apiMethod}`);
      }
    } else {
      throw new Error('Either apiCall function or apiClient with apiEndpoint must be provided');
    }

    // Extract data based on dataPath
    let data = response.data;
    if (dataPath) {
      const pathParts = dataPath.split('.');
      for (const part of pathParts) {
        if (data && typeof data === 'object') {
          data = data[part];
        } else {
          data = undefined;
          break;
        }
      }
    }

    // Call success callback
    if (onSuccess) {
      onSuccess(data, response);
    }

    // Show success notification
    if (showSuccess && successMessage && notify) {
      notify.success(successMessage);
    }

    return { success: true, data, response };
  } catch (error) {
    // Call error callback
    if (onError) {
      onError(error);
    }

    // Handle error
    if (handleError) {
      handleError(error, errorContext);
    }

    // Show error notification
    if (showError && errorMessage && notify) {
      notify.error(errorMessage);
    }

    // Return fallback value if provided
    if (fallbackValue !== undefined) {
      return { success: false, data: fallbackValue, error };
    }

    return { success: false, error };
  }
};

export const performDataOperation = async (operationId, options) => {
  const {
    apiClient,
    apiMethod = 'post',
    apiEndpoint,
    requestData,
    onBeforeStart,
    onSuccess,
    onError,
    onFinally,
    successMessage,
    errorMessage,
    errorContext,
    showSuccess = true,
    showError = true,
    notify,
    handleError,
    apiCall // Support apiCall parameter for direct API function calls
  } = options;

  try {
    // Call before start callback
    if (onBeforeStart) {
      onBeforeStart();
    }

    // Make API call - support both apiCall function and traditional apiClient + apiEndpoint
    let response;
    if (apiCall && typeof apiCall === 'function') {
      // Use apiCall function if provided
      response = await apiCall();
    } else if (apiClient && apiEndpoint) {
      // Traditional apiClient + apiEndpoint approach
      if (apiMethod === 'post') {
        response = await apiClient.post(apiEndpoint, requestData);
      } else if (apiMethod === 'put') {
        response = await apiClient.put(apiEndpoint, requestData);
      } else if (apiMethod === 'delete') {
        response = await apiClient.delete(apiEndpoint, requestData ? { data: requestData } : undefined);
      } else if (apiMethod === 'get') {
        response = await apiClient.get(apiEndpoint, requestData ? { params: requestData } : undefined);
      } else {
        throw new Error(`Unsupported API method: ${apiMethod}`);
      }
    } else {
      throw new Error('Either apiCall function or apiClient with apiEndpoint must be provided');
    }

    // Call success callback
    if (onSuccess) {
      onSuccess(response.data, response);
    }

    // Show success notification
    if (showSuccess && successMessage && notify) {
      notify.success(successMessage);
    }

    return { success: true, data: response.data, response };
  } catch (error) {
    // Call error callback
    if (onError) {
      onError(error);
    }

    // Handle error
    if (handleError) {
      handleError(error, errorContext);
    }

    // Show error notification
    if (showError && errorMessage && notify) {
      notify.error(errorMessage);
    }

    return { success: false, error };
  } finally {
    // Call finally callback
    if (onFinally) {
      onFinally();
    }
  }
};

export default {
  performDataLoad,
  performDataOperation
};
