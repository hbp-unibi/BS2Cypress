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
	EXPECT_EQ(bs_pops[0]->size(), size_t(20));
	EXPECT_EQ(bs_pops[1]->size(), size_t(10));
	EXPECT_EQ(bs_pops[2]->size(), size_t(30));
	EXPECT_TRUE(bs_pops[3] == nullptr);
}

TEST(BrainScaleS, set_BS_params)
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
	BrainScaleS::set_BS_params(params, src);
	EXPECT_FLOAT_EQ(params.cm, src.cm());
	EXPECT_FLOAT_EQ(params.v_reset, src.v_reset());
	EXPECT_FLOAT_EQ(params.v_thresh, src.v_thresh());
	EXPECT_FLOAT_EQ(params.e_rev_E, src.e_rev_E());
	EXPECT_FLOAT_EQ(params.tau_m, src.tau_m());
	EXPECT_FLOAT_EQ(params.tau_refrac, src.tau_refrac());
	EXPECT_FLOAT_EQ(params.tau_syn_E, src.tau_syn_E());
	EXPECT_FLOAT_EQ(params.tau_syn_I, src.tau_syn_I());
}

TEST(BrainScaleS, set_BS_params_2)
{
	CellTypeTraits<CellType::EIF_cond_exp_isfa_ista>::Parameters params;
	auto src = EifCondExpIsfaIstaParameters()
	               .cm(1)
	               .v_reset(2)
	               .v_rest(3)
	               .v_thresh(5)
	               .e_rev_E(4)
	               .tau_m(5)
	               .tau_refrac(8)
	               .tau_syn_E(-21)
	               .tau_syn_I(-50)
	               .tau_w(29)
	               .a(821)
	               .b(49)
	               .delta_T(3);
	BrainScaleS::set_BS_params(params, src);
	EXPECT_FLOAT_EQ(params.cm, src.cm());
	EXPECT_FLOAT_EQ(params.v_reset, src.v_reset());
	EXPECT_FLOAT_EQ(params.v_thresh, src.v_thresh());
	EXPECT_FLOAT_EQ(params.e_rev_E, src.e_rev_E());
	EXPECT_FLOAT_EQ(params.tau_m, src.tau_m());
	EXPECT_FLOAT_EQ(params.tau_refrac, src.tau_refrac());
	EXPECT_FLOAT_EQ(params.tau_syn_E, src.tau_syn_E());
	EXPECT_FLOAT_EQ(params.tau_syn_I, src.tau_syn_I());

	EXPECT_FLOAT_EQ(params.tau_w, src.tau_w());
	EXPECT_FLOAT_EQ(params.a, src.a());
	EXPECT_FLOAT_EQ(params.b, src.b());
	EXPECT_FLOAT_EQ(params.delta_T, src.delta_T());
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

	TypedCellParameterVector<CellType::EIF_cond_exp_isfa_ista> params_vec2(10);
	auto src2 = EifCondExpIsfaIstaParameters()
	                .cm(1)
	                .v_reset(2)
	                .v_rest(3)
	                .v_thresh(5)
	                .e_rev_E(4)
	                .tau_m(5)
	                .tau_refrac(8)
	                .tau_syn_E(-21)
	                .tau_syn_I(-50)
	                .tau_w(29)
	                .a(821)
	                .b(49)
	                .delta_T(3);
	BrainScaleS::set_hom_param(params_vec2, src2);
	for (auto i : params_vec2.parameters()) {
		EXPECT_FLOAT_EQ(i.cm, src2.cm());
		EXPECT_FLOAT_EQ(i.v_reset, src2.v_reset());
		EXPECT_FLOAT_EQ(i.v_thresh, src2.v_thresh());
		EXPECT_FLOAT_EQ(i.e_rev_E, src2.e_rev_E());
		EXPECT_FLOAT_EQ(i.tau_m, src2.tau_m());
		EXPECT_FLOAT_EQ(i.tau_refrac, src2.tau_refrac());
		EXPECT_FLOAT_EQ(i.tau_syn_E, src2.tau_syn_E());
		EXPECT_FLOAT_EQ(i.tau_syn_I, src2.tau_syn_I());

		EXPECT_FLOAT_EQ(i.tau_w, src2.tau_w());
		EXPECT_FLOAT_EQ(i.a, src2.a());
		EXPECT_FLOAT_EQ(i.b, src2.b());
		EXPECT_FLOAT_EQ(i.delta_T, src2.delta_T());
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

	TypedCellParameterVector<CellType::EIF_cond_exp_isfa_ista> params_vec2(10);
	auto pop2 = netw.create_population<EifCondExpIsfaIsta>(
	    10, EifCondExpIsfaIstaParameters());
	for (size_t i = 0; i < 10; i++) {
		pop2[i].parameters() = EifCondExpIsfaIstaParameters()
		                           .cm(1 + 2 * i)
		                           .v_reset(2 + 9 * i)
		                           .v_rest(3 - 28 * i)
		                           .v_thresh(5 - 4 * i)
		                           .e_rev_E(4 + 2 * i)
		                           .tau_m(5 * i)
		                           .tau_refrac(i * 8 + 3)
		                           .tau_syn_E(-21 * i)
		                           .tau_syn_I(-50 * i * i)
		                           .tau_w(29 - i)
		                           .a(821 - 2 * i)
		                           .b(49 + i * 3)
		                           .delta_T(3 * i);
	}
	BrainScaleS::set_inhom_param(params_vec2, pop2);
	auto params2 = params_vec2.parameters();
	for (size_t i = 0; i < 10; i++) {
		EXPECT_FLOAT_EQ(params2[i].cm, pop2[i].parameters().cm());
		EXPECT_FLOAT_EQ(params2[i].v_reset, pop2[i].parameters().v_reset());
		EXPECT_FLOAT_EQ(params2[i].v_thresh, pop2[i].parameters().v_thresh());
		EXPECT_FLOAT_EQ(params2[i].e_rev_E, pop2[i].parameters().e_rev_E());
		EXPECT_FLOAT_EQ(params2[i].tau_m, pop2[i].parameters().tau_m());
		EXPECT_FLOAT_EQ(params2[i].tau_refrac,
		                pop2[i].parameters().tau_refrac());
		EXPECT_FLOAT_EQ(params2[i].tau_syn_E, pop2[i].parameters().tau_syn_E());
		EXPECT_FLOAT_EQ(params2[i].tau_syn_I, pop2[i].parameters().tau_syn_I());

		EXPECT_FLOAT_EQ(params2[i].tau_w, pop2[i].parameters().tau_w());
		EXPECT_FLOAT_EQ(params2[i].a, pop2[i].parameters().a());
		EXPECT_FLOAT_EQ(params2[i].b, pop2[i].parameters().b());
		EXPECT_FLOAT_EQ(params2[i].delta_T, pop2[i].parameters().delta_T());
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

	conn_desc =
	    ConnectionDescriptor(0, 0, 16, 1, 0, 16, Connector::from_list(vec));
	conn = BrainScaleS::get_connector(conn_desc);

	EXPECT_FALSE(conn);
}

TEST(BrainScaleS, get_list_connector)
{
	std::vector<LocalConnection> vec;

	vec.emplace_back(LocalConnection{0, 0, 1.0, 1.1});
	vec.emplace_back(LocalConnection{0, 1, 0.9, 1.2});
	vec.emplace_back(LocalConnection{0, 2, 0.8, 1.3});
	vec.emplace_back(LocalConnection{0, 3, 0.7, 1.4});
	vec.emplace_back(LocalConnection{0, 4, 0.6, 1.5});
	vec.emplace_back(LocalConnection{0, 5, 0.5, 1.6});
	vec.emplace_back(LocalConnection{0, 6, -0.5, 1.7});
	vec.emplace_back(LocalConnection{0, 7, -0.4, 1.8});
	vec.emplace_back(LocalConnection{0, 8, -0.3, 1.9});
	vec.emplace_back(LocalConnection{0, 9, -0.2, 1.95});
	vec.emplace_back(LocalConnection{0, 10, -0.1, 2.0});
	vec.emplace_back(LocalConnection{0, 11, -0.15, 2.5});

	auto conn_desc =
	    ConnectionDescriptor(0, 0, 16, 1, 0, 16, Connector::from_list(vec));
	std::vector<cypress::LocalConnection> conns;
	auto conn = BrainScaleS::get_list_connector(conn_desc, conns);

	auto weights = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<0>(conn)->getDefaultWeights());
	EXPECT_EQ(weights.size(), size_t(6));
	EXPECT_FLOAT_EQ(weights[0], 1.0);
	EXPECT_FLOAT_EQ(weights[1], 0.9);
	EXPECT_FLOAT_EQ(weights[2], 0.8);
	EXPECT_FLOAT_EQ(weights[3], 0.7);
	EXPECT_FLOAT_EQ(weights[4], 0.6);
	EXPECT_FLOAT_EQ(weights[5], 0.5);

	auto delays = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<0>(conn)->getDefaultDelays());
	EXPECT_EQ(delays.size(), size_t(6));
	EXPECT_FLOAT_EQ(delays[0], 1.1);
	EXPECT_FLOAT_EQ(delays[1], 1.2);
	EXPECT_FLOAT_EQ(delays[2], 1.3);
	EXPECT_FLOAT_EQ(delays[3], 1.4);
	EXPECT_FLOAT_EQ(delays[4], 1.5);
	EXPECT_FLOAT_EQ(delays[5], 1.6);

	weights = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<1>(conn)->getDefaultWeights());
	EXPECT_EQ(weights.size(), size_t(6));
	EXPECT_FLOAT_EQ(weights[0], 0.5);
	EXPECT_FLOAT_EQ(weights[1], 0.4);
	EXPECT_FLOAT_EQ(weights[2], 0.3);
	EXPECT_FLOAT_EQ(weights[3], 0.2);
	EXPECT_FLOAT_EQ(weights[4], 0.1);
	EXPECT_FLOAT_EQ(weights[5], 0.15);

	delays = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<1>(conn)->getDefaultDelays());
	EXPECT_EQ(delays.size(), size_t(6));
	EXPECT_FLOAT_EQ(delays[0], 1.7);
	EXPECT_FLOAT_EQ(delays[1], 1.8);
	EXPECT_FLOAT_EQ(delays[2], 1.9);
	EXPECT_FLOAT_EQ(delays[3], 1.95);
	EXPECT_FLOAT_EQ(delays[4], 2.0);
	EXPECT_FLOAT_EQ(delays[5], 2.5);

	/* =======================================================================*/
	vec.clear();
	vec.emplace_back(LocalConnection{0, 0, 1.0, 1.1});
	vec.emplace_back(LocalConnection{0, 1, 0.9, 1.2});
	vec.emplace_back(LocalConnection{0, 2, 0.8, 1.3});
	vec.emplace_back(LocalConnection{0, 3, 0.7, 1.4});
	vec.emplace_back(LocalConnection{0, 4, 0.6, 1.5});
	vec.emplace_back(LocalConnection{0, 5, 0.5, 1.6});

	conn_desc =
	    ConnectionDescriptor(0, 0, 16, 1, 0, 16, Connector::from_list(vec));
	conns.clear();
	conn = BrainScaleS::get_list_connector(conn_desc, conns);

	weights = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<0>(conn)->getDefaultWeights());
	EXPECT_EQ(weights.size(), size_t(6));
	EXPECT_FLOAT_EQ(weights[0], 1.0);
	EXPECT_FLOAT_EQ(weights[1], 0.9);
	EXPECT_FLOAT_EQ(weights[2], 0.8);
	EXPECT_FLOAT_EQ(weights[3], 0.7);
	EXPECT_FLOAT_EQ(weights[4], 0.6);
	EXPECT_FLOAT_EQ(weights[5], 0.5);

	delays = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<0>(conn)->getDefaultDelays());
	EXPECT_EQ(delays.size(), size_t(6));
	EXPECT_FLOAT_EQ(delays[0], 1.1);
	EXPECT_FLOAT_EQ(delays[1], 1.2);
	EXPECT_FLOAT_EQ(delays[2], 1.3);
	EXPECT_FLOAT_EQ(delays[3], 1.4);
	EXPECT_FLOAT_EQ(delays[4], 1.5);
	EXPECT_FLOAT_EQ(delays[5], 1.6);

	weights = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<1>(conn)->getDefaultWeights());
	EXPECT_EQ(weights.size(), size_t(0));
	delays = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<1>(conn)->getDefaultDelays());
	EXPECT_EQ(delays.size(), size_t(0));

	/* =======================================================================*/
	vec.clear();
	vec.emplace_back(LocalConnection{0, 6, -0.5, 1.7});
	vec.emplace_back(LocalConnection{0, 7, -0.4, 1.8});
	vec.emplace_back(LocalConnection{0, 8, -0.3, 1.9});
	vec.emplace_back(LocalConnection{0, 9, -0.2, 1.95});
	vec.emplace_back(LocalConnection{0, 10, -0.1, 2.0});
	vec.emplace_back(LocalConnection{0, 11, -0.15, 2.5});

	conn_desc =
	    ConnectionDescriptor(0, 0, 16, 1, 0, 16, Connector::from_list(vec));
	conns.clear();
	conn = BrainScaleS::get_list_connector(conn_desc, conns);

	weights = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<0>(conn)->getDefaultWeights());
	EXPECT_EQ(weights.size(), size_t(0));

	delays = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<0>(conn)->getDefaultDelays());
	EXPECT_EQ(delays.size(), size_t(0));

	weights = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<1>(conn)->getDefaultWeights());
	EXPECT_EQ(weights.size(), size_t(6));
	EXPECT_FLOAT_EQ(weights[0], 0.5);
	EXPECT_FLOAT_EQ(weights[1], 0.4);
	EXPECT_FLOAT_EQ(weights[2], 0.3);
	EXPECT_FLOAT_EQ(weights[3], 0.2);
	EXPECT_FLOAT_EQ(weights[4], 0.1);
	EXPECT_FLOAT_EQ(weights[5], 0.15);

	delays = boost::get<boost::numeric::ublas::vector<float>>(
	    std::get<1>(conn)->getDefaultDelays());
	EXPECT_EQ(delays.size(), size_t(6));
	EXPECT_FLOAT_EQ(delays[0], 1.7);
	EXPECT_FLOAT_EQ(delays[1], 1.8);
	EXPECT_FLOAT_EQ(delays[2], 1.9);
	EXPECT_FLOAT_EQ(delays[3], 1.95);
	EXPECT_FLOAT_EQ(delays[4], 2.0);
	EXPECT_FLOAT_EQ(delays[5], 2.5);
}

TEST(BrainScaleS, get_popview)
{
	ObjectStore store;
	auto pop = ::Population::create(store, 10, CellType::IF_cond_exp);

	auto view = BrainScaleS::get_popview(pop, 0, 10);
	EXPECT_EQ(view.size(), size_t(10));

	view = BrainScaleS::get_popview(pop, 0, 11);
	EXPECT_EQ(view.size(), size_t(10));

	view = BrainScaleS::get_popview(pop, 3, 4);
	EXPECT_EQ(view.size(), size_t(1));
	view = BrainScaleS::get_popview(pop, 3, 7);
	EXPECT_EQ(view.size(), size_t(4));
	view = BrainScaleS::get_popview(pop, 10, 11);
	EXPECT_EQ(view.size(), size_t(0));
	view = BrainScaleS::get_popview(pop, 1, 1);
	EXPECT_EQ(view.size(), size_t(0));
	view = BrainScaleS::get_popview(pop, 9, 7);
	EXPECT_EQ(view.size(), size_t(0));
}

TEST(BrainScaleS, fetch_data)
{
	StaticSynapse synapse = StaticSynapse().weight(15).delay(1);
	Json json(
	    {{"digital_weight", true}, {"neuron_size", 16} /*, {"ess", true}*/});
	// TODO awaiting bugfix
	auto backend = BrainScaleS(json);

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
	                .v_rest(-50)
	                .v_thresh(-30)
	                .e_rev_E(60)
	                .tau_m(20)
	                .tau_refrac(0.1)
	                .tau_syn_E(5.0),
	            IfCondExpSignals().record_spikes().record_v())
	        .add_connection("source", "target", Connector::all_to_all(synapse))
	        .run(backend, 800.0);

	auto pop = net.population<IfCondExp>("target");

	std::cout
	    << "NOTE: The following test results underlie statistical variations "
	       "and should be interpreted with care. A deviation might not "
	       "necessarily be the result of a broken implementation!"
	    << std::endl;

	size_t size = pop[0].signals().get_spikes().size();
	EXPECT_TRUE(size > 0);
	EXPECT_NEAR(pop[0].signals().get_spikes()[0], 251.0, 5);
	EXPECT_NEAR(pop[0].signals().get_spikes()[size - 1], 507.0, 5);

	auto v_and_time = pop[0].signals().get_v();
	size = v_and_time.rows();
	EXPECT_NEAR(v_and_time(0, 0), 0.1, 0.1);
	EXPECT_NEAR(v_and_time(15, 1), -50.0, 10.0);

	EXPECT_NEAR(v_and_time(size - 1, 0), 800.0, 0.1);
	EXPECT_NEAR(v_and_time(size - 1, 1), -50.0, 10.0);
}

TEST(BrainScaleS, low_level_from_list)
{
	cypress::global_logger().min_level(LogSeverity::WARNING);
	Json json(
	    {{"digital_weight", true}, {"neuron_size", 16} /*, {"ess", true}*/});
	// TODO awaiting bugfix
	auto backend = BrainScaleS(json);

	size_t max_weight = 15;
	auto net =
	    Network()
	        // Add a named population of poisson spike sources
	        .add_population<SpikeSourceConstFreq>(
	            "source", 1,
	            SpikeSourceConstFreqParameters()
	                .start(5.0)
	                .rate(100.0)
	                .duration(5000.0),
	            SpikeSourceConstFreqSignals())
	        .add_population<IfCondExp>("target", max_weight,
	                                   IfCondExpParameters()
	                                       .cm(0.2)
	                                       .v_reset(-70)
	                                       .v_rest(-60)
	                                       .v_thresh(-45)
	                                       .e_rev_E(60)
	                                       .tau_m(20)
	                                       .tau_refrac(1.0)
	                                       .tau_syn_E(10.0),
	                                   IfCondExpSignals().record_spikes());
	/*.add_population<IfCondExp>("target2", 1,
	                           IfCondExpParameters()
	                               .cm(0.2)
	                               .v_reset(-70)
	                               .v_rest(-20)
	                               .v_thresh(-15)
	                               .e_rev_E(60)
	                               .tau_m(20)
	                               .tau_refrac(0.1)
	                               .tau_syn_E(3.0),
	                           IfCondExpSignals().record_spikes());*/

	std::vector<LocalConnection> conns;
	for (size_t i = 0; i < max_weight; i++) {
		conns.emplace_back(LocalConnection(0, i, i, 1.0));
	}
	net.add_connection("source", "target", Connector::from_list(conns));
	net.run(backend, 5010);

	auto pop = net.population("target");
	for (size_t i = 1; i < max_weight; i++) {
		std::cout << pop[i].signals().data(0).size() << " bigger ? "
		          << pop[i-1].signals().data(0).size() << std::endl;
		EXPECT_TRUE(pop[i].signals().data(0).size() >
		            pop[i - 1].signals().data(0).size());
	}
}

}  // namespace cypress
