#include "JoinQuery.hpp"
#include <assert.h>
#include <fstream>
#include <thread>
//---------------------------------------------------------------------------
JoinQuery::JoinQuery(std::string lineitem, std::string order,
                     std::string customer)
{}
//---------------------------------------------------------------------------
size_t JoinQuery::avg(std::string segmentParam)
{
   return 1;
}
//---------------------------------------------------------------------------
size_t JoinQuery::lineCount(std::string rel)
{
   std::ifstream relation(rel);
   assert(relation);  // make sure the provided string references a file
   size_t n = 0;
   for (std::string line; std::getline(relation, line);) n++;
   return n;
}
//---------------------------------------------------------------------------
