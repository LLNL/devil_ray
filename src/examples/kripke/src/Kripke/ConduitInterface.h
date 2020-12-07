//
// Copyright (c) 2014-19, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#ifndef KRIPKE_CONDUIT_INTERFACE_H__
#define KRIPKE_CONDUIT_INTERFACE_H__

#include <Kripke.h>
#include <Kripke/Core/DataStore.h>
#include <Kripke/VarTypes.h>
#include <vector>
#include <conduit.hpp>

namespace Kripke {

  class DataStore;

  void ToBlueprint(Kripke::Core::DataStore &data_store,
                   conduit::Node &dataset);

} // namespace

#endif

