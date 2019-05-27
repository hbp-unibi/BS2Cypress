/*
 *  Cypress -- C++ Spiking Neural Network Simulation Framework
 *  Copyright (C) 2019 Christoph Ostrau
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wregister"
#include <cypress/cypress.hpp>
#pragma GCC diagnostic pop

#include "backend/brainscales.hpp"

#include "gtest/gtest.h"

#include <sstream>

namespace cypress {

TEST(BrainScaleS, init_logger) { EXPECT_NO_THROW(BrainScaleS::init_logger()); }

TEST(BrainScaleS, create_pops)
{
	std::vector<PopulationBase> pops;
	Network netw;
	pops.emplace_back(
	    netw.create_population<IfCondExp>(20, IfCondExpParameters()));
	pops.emplace_back(
	    netw.create_population<IfCondExp>(10, IfCondExpParameters()));
	pops.emplace_back(
	    netw.create_population<IfCondExp>(30, IfCondExpParameters()));
	pops.emplace_back(
	    netw.create_population<IfCondExp>(00, IfCondExpParameters()));
	ObjectStore store;
	auto bs_pops = BrainScaleS::create_pops(store, pops);
	EXPECT_EQ(bs_pops[0]->size(), 20);
	EXPECT_EQ(bs_pops[1]->size(), 10);
	EXPECT_EQ(bs_pops[2]->size(), 30);
	EXPECT_TRUE(bs_pops[3] == nullptr);
}

TEST(BrainScaleS, set_ifce_params)
{
	CellTypeTraits<CellType::IF_cond_exp>::Parameters params;
	auto src = IfCondExpParameters()
	               .cm(1)
	               .v_reset(2)
	               .v_rest(3)
	               .v_thresh(5)
	               .e_rev_E(4)
	               .tau_m(5)
	               .tau_refrac(8)
	               .tau_syn_E(-21)
	               .tau_syn_I(-50);
	BrainScaleS::set_ifce_params(params, src);
	EXPECT_FLOAT_EQ(params.cm, src.cm());
	EXPECT_FLOAT_EQ(params.v_reset, src.v_reset());
	EXPECT_FLOAT_EQ(params.v_thresh, src.v_thresh());
	EXPECT_FLOAT_EQ(params.e_rev_E, src.e_rev_E());
	EXPECT_FLOAT_EQ(params.tau_m, src.tau_m());
	EXPECT_FLOAT_EQ(params.tau_refrac, src.tau_refrac());
	EXPECT_FLOAT_EQ(params.tau_syn_E, src.tau_syn_E());
	EXPECT_FLOAT_EQ(params.tau_syn_I, src.tau_syn_I());
}

TEST(BrainScaleS, set_hom_param)
{
	TypedCellParameterVector<CellType::IF_cond_exp> params_vec(10);
	auto src = IfCondExpParameters()
	               .cm(1)
	               .v_reset(2)
	               .v_rest(3)
	               .v_thresh(5)
	               .e_rev_E(4)
	               .tau_m(5)
	               .tau_refrac(8)
	               .tau_syn_E(-21)
	               .tau_syn_I(-50);
	BrainScaleS::set_hom_param(params_vec, src);
	for (auto i : params_vec.parameters()) {
		EXPECT_FLOAT_EQ(i.cm, src.cm());
		EXPECT_FLOAT_EQ(i.v_reset, src.v_reset());
		EXPECT_FLOAT_EQ(i.v_thresh, src.v_thresh());
		EXPECT_FLOAT_EQ(i.e_rev_E, src.e_rev_E());
		EXPECT_FLOAT_EQ(i.tau_m, src.tau_m());
		EXPECT_FLOAT_EQ(i.tau_refrac, src.tau_refrac());
		EXPECT_FLOAT_EQ(i.tau_syn_E, src.tau_syn_E());
		EXPECT_FLOAT_EQ(i.tau_syn_I, src.tau_syn_I());
	}
}

TEST(BrainScaleS, set_inhom_param)
{
	TypedCellParameterVector<CellType::IF_cond_exp> params_vec(10);
	Network netw;
	auto pop = netw.create_population<IfCondExp>(10, IfCondExpParameters());
	for (size_t i = 0; i < 10; i++) {
		pop[i].parameters() = IfCondExpParameters()
		                          .cm(1 + 2 * i)
		                          .v_reset(2 + 9 * i)
		                          .v_rest(3 - 28 * i)
		                          .v_thresh(5 - 4 * i)
		                          .e_rev_E(4 + 2 * i)
		                          .tau_m(5 * i)
		                          .tau_refrac(i * 8 + 3)
		                          .tau_syn_E(-21 * i)
		                          .tau_syn_I(-50 * i * i);
	}
	BrainScaleS::set_inhom_param(params_vec, pop);
	auto params = params_vec.parameters();
	for (size_t i = 0; i < 10; i++) {
		EXPECT_FLOAT_EQ(params[i].cm, pop[i].parameters().cm());
		EXPECT_FLOAT_EQ(params[i].v_reset, pop[i].parameters().v_reset());
		EXPECT_FLOAT_EQ(params[i].v_thresh, pop[i].parameters().v_thresh());
		EXPECT_FLOAT_EQ(params[i].e_rev_E, pop[i].parameters().e_rev_E());
		EXPECT_FLOAT_EQ(params[i].tau_m, pop[i].parameters().tau_m());
		EXPECT_FLOAT_EQ(params[i].tau_refrac, pop[i].parameters().tau_refrac());
		EXPECT_FLOAT_EQ(params[i].tau_syn_E, pop[i].parameters().tau_syn_E());
		EXPECT_FLOAT_EQ(params[i].tau_syn_I, pop[i].parameters().tau_syn_I());
	}
}

TEST(BrainScaleS, set_population_parameters) {}

TEST(BrainScaleS, set_hom_rec)
{
	Network netw;
	auto pop = netw.create_population<IfCondExp>(
	    10, IfCondExpParameters(), IfCondExpSignals().record_spikes());
	TypedCellParameterVector<CellType::IF_cond_exp> params_vec(10);

	auto params = params_vec.parameters();
	BrainScaleS::set_hom_rec(params, pop);

	for (size_t i = 0; i < 10; i++) {
		EXPECT_TRUE(params[i].record_spikes);
		EXPECT_FALSE(params[i].record_v);
		EXPECT_FALSE(params[i].record_gsyn);
	}

	auto pop2 = netw.create_population<IfCondExp>(
	    10, IfCondExpParameters(),
	    IfCondExpSignals().record_spikes().record_v().record_gsyn_exc());
	TypedCellParameterVector<CellType::IF_cond_exp> params_vec2(10);

	auto params2 = params_vec2.parameters();
	BrainScaleS::set_hom_rec(params2, pop2);

	for (size_t i = 0; i < 10; i++) {
		EXPECT_TRUE(params2[i].record_spikes);
		EXPECT_TRUE(params2[i].record_v);
		EXPECT_TRUE(params2[i].record_gsyn);
	}
}

TEST(BrainScaleS, set_inhom_rec)
{
	Network netw;
	auto pop = netw.create_population<IfCondExp>(10, IfCondExpParameters(),
	                                             IfCondExpSignals());
	pop[5].signals().record_spikes();
	pop[6].signals().record_v();
	pop[7].signals().record_gsyn_exc();
	TypedCellParameterVector<CellType::IF_cond_exp> params_vec(10);

	auto params = params_vec.parameters();
	BrainScaleS::set_inhom_rec(params, pop);

	for (size_t i = 0; i < 10; i++) {
		if (i == 5) {
			EXPECT_TRUE(params[i].record_spikes);
		}
		else {
			EXPECT_FALSE(params[i].record_spikes);
		}
		if (i == 6) {
			EXPECT_TRUE(params[i].record_v);
		}
		else {
			EXPECT_FALSE(params[i].record_v);
		}
		if (i == 7) {
			EXPECT_TRUE(params[i].record_gsyn);
		}
		else {
			EXPECT_FALSE(params[i].record_gsyn);
		}
	}
}

TEST(BrainScaleS, set_hom_rec_spikes)
{
	Network netw;
	TypedCellParameterVector<CellType::SpikeSourceArray> params_vec(10);

	auto params = params_vec.parameters();
	BrainScaleS::set_hom_rec_spikes(params);

	for (size_t i = 0; i < 10; i++) {
		EXPECT_TRUE(params[i].record_spikes);
	}
}

TEST(BrainScaleS, set_inhom_rec_spikes)
{
	Network netw;
	auto pop = netw.create_population<SpikeSourceArray>(
	    10, SpikeSourceArrayParameters());
	pop[5].signals().record_spikes();
	pop[7].signals().record_spikes();
	TypedCellParameterVector<CellType::SpikeSourceArray> params_vec(10);

	auto params = params_vec.parameters();
	BrainScaleS::set_inhom_rec_spikes(params, pop);

	for (size_t i = 0; i < 10; i++) {
		if ((i == 5) || (i == 7)) {
			EXPECT_TRUE(params[i].record_spikes);
		}
		else {
			EXPECT_FALSE(params[i].record_spikes);
		}
	}
}

TEST(BrainScaleS, set_population_records) {}

TEST(BrainScaleS, get_connector)
{
	ConnectionDescriptor conn_desc(0, 0, 16, 1, 0, 16,
	                               Connector::all_to_all(0.15, 2.5));
	auto conn = BrainScaleS::get_connector(conn_desc);
	auto weight = boost::get<float>(conn->getDefaultWeights());
	auto delay = boost::get<float>(conn->getDefaultDelays());
	EXPECT_FLOAT_EQ(weight, 0.15);
	EXPECT_FLOAT_EQ(delay, 2.5);

	std::vector<LocalConnection> vec;

	vec.emplace_back(LocalConnection{0, 0, 1.0, 1.1});
	vec.emplace_back(LocalConnection{0, 1, 0.9, 1.2});
	vec.emplace_back(LocalConnection{0, 2, 0.8, 1.3});
	vec.emplace_back(LocalConnection{0, 3, 0.7, 1.4});
	vec.emplace_back(LocalConnection{0, 4, 0.6, 1.5});
	vec.emplace_back(LocalConnection{0, 5, 0.5, 1.6});

	conn_desc =
	    ConnectionDescriptor(0, 0, 16, 1, 0, 16, Connector::from_list(vec));
	conn = BrainScaleS::get_connector(conn_desc);

	auto weights = boost::get<boost::numeric::ublas::vector<float>>(
	    conn->getDefaultWeights());
	EXPECT_FLOAT_EQ(weights[0], 1.0);
	EXPECT_FLOAT_EQ(weights[1], 0.9);
	EXPECT_FLOAT_EQ(weights[2], 0.8);
	EXPECT_FLOAT_EQ(weights[3], 0.7);
	EXPECT_FLOAT_EQ(weights[4], 0.6);
	EXPECT_FLOAT_EQ(weights[5], 0.5);

	auto delays = boost::get<boost::numeric::ublas::vector<float>>(
	    conn->getDefaultDelays());
	EXPECT_FLOAT_EQ(delays[0], 1.1);
	EXPECT_FLOAT_EQ(delays[1], 1.2);
	EXPECT_FLOAT_EQ(delays[2], 1.3);
	EXPECT_FLOAT_EQ(delays[3], 1.4);
	EXPECT_FLOAT_EQ(delays[4], 1.5);
	EXPECT_FLOAT_EQ(delays[5], 1.6);
}

TEST(BrainScaleS, get_popview)
{
	ObjectStore store;
	auto pop = ::Population::create(store, 10, CellType::IF_cond_exp);

	auto view = BrainScaleS::get_popview(pop, 0, 10);
	EXPECT_EQ(view.size(), 10);

	view = BrainScaleS::get_popview(pop, 0, 11);
	EXPECT_EQ(view.size(), 10);

	view = BrainScaleS::get_popview(pop, 3, 4);
	EXPECT_EQ(view.size(), 1);
	view = BrainScaleS::get_popview(pop, 3, 7);
	EXPECT_EQ(view.size(), 4);
	view = BrainScaleS::get_popview(pop, 10, 11);
	EXPECT_EQ(view.size(), 0);
	view = BrainScaleS::get_popview(pop, 1, 1);
	EXPECT_EQ(view.size(), 0);
	view = BrainScaleS::get_popview(pop, 9, 7);
	EXPECT_EQ(view.size(), 0);
}

TEST(BrainScaleS, fetch_data)
{
	StaticSynapse synapse = StaticSynapse().weight(15).delay(1);
	auto net =
	    Network()
	        // Add a named population of poisson spike sources
	        .add_population<SpikeSourceArray>(
	            "source", 1, SpikeSourceArrayParameters({250, 500}),
	            SpikeSourceArraySignals().record_spikes())
	        .add_population<IfCondExp>(
	            "target", 1,
	            IfCondExpParameters()
	                .cm(0.2)
	                .v_reset(-70)
	                .v_rest(-20)
	                .v_thresh(-10)
	                .e_rev_E(60)
	                .tau_m(20)
	                .tau_refrac(0.1)
	                .tau_syn_E(5.0),
	            IfCondExpSignals().record_spikes().record_v())
	        .add_connection("source", "target", Connector::all_to_all(synapse))
	        .run("nmpm1", 600.0);

	auto pop = net.population<IfCondExp>("target");

	std::cout
	    << "NOTE: The following test results underlie statistical variations "
	       "and should be interpreted with care. A deviation might not "
	       "necessarily be the result of a broken implementation!"
	    << std::endl;

	size_t size = pop[0].signals().get_spikes().size();
	EXPECT_NEAR(pop[0].signals().get_spikes()[0], 251.0, 5);
	EXPECT_NEAR(pop[0].signals().get_spikes()[size - 1], 507.0, 5);

	auto v_and_time = pop[0].signals().get_v();
	size = v_and_time.rows();
	EXPECT_NEAR(v_and_time(0, 0), 0.1, 0.1);
	EXPECT_NEAR(v_and_time(0, 1), -20.0, 10.0);

	EXPECT_NEAR(v_and_time(size - 1, 0), 600.0, 0.1);
	EXPECT_NEAR(v_and_time(size - 1, 1), -20.0, 10.0);
}

}  // namespace cypress
