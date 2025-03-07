#ifndef __TESTS_ALL_IN_ONE_FS_UTIL_H
#define __TESTS_ALL_IN_ONE_FS_UTIL_H


#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

/**
 * @brief Get the absolute path of files in the directory 
 * 
 * @param directory 
 * @return std::vector<fs::path> 
 */
std::vector<fs::path> get_files_in_directory(const fs::path& directory);



#endif