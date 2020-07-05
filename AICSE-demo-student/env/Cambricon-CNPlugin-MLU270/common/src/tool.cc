#include "tool.h"
#include <vector>
#include <fstream>


std::vector<std::vector<float>> CSVReader::getData()
{
  std::ifstream file(fileName);

  std::vector<std::vector<float>> dataList;

  std::string line;
  // Iterate through each line and split the content using delimeter
  while (getline(file, line))
  {
    std::vector<float> vec;
    std::string item;
    std::istringstream in(line);
    while (getline(in, item, ','))
    {
      vec.push_back(atof(item.c_str()));
    }
    dataList.push_back(vec);
  }
  // Close the File
  file.close();

  return dataList;
}